# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCState end


sample_available(state::AbstractMCMCState) = sample_available(state, Val(:complete))

current_sample(state::AbstractMCMCState) = current_sample(state, Val(:complete))


function Base.push!(xs::DensitySampleVector, state::AbstractMCMCState)
    if sample_available(state, Val(:any))
        push!(xs, current_sample(state, Val(:any)))
    end
    xs
end



abstract type MCMCAlgorithm{S<:AbstractMCMCState} <: BATAlgorithm end



mcmc_compatible(::MCMCAlgorithm, ::AbstractProposalDist, ::AbstractParamBounds) = true

function sample_weight_type end


rand_initial_params(rng::AbstractRNG, algorithm::MCMCAlgorithm, target::TargetSubject) =
    rand_initial_params!(rng, algorithm, target, Vector{float(eltype(target.bounds))}(nparams(target)))

rand_initial_params(rng::AbstractRNG, algorithm::MCMCAlgorithm, target::TargetSubject, n::Integer) =
    rand_initial_params!(rng, algorithm, target, Matrix{float(eltype(target.bounds))}(nparams(target), n))

rand_initial_params!(rng::AbstractRNG, ::MCMCAlgorithm, target::TargetSubject, x::StridedVecOrMat{<:Real}) =
    rand!(rng, target.bounds, x)



mutable struct MCMCIterator{
    A<:MCMCAlgorithm,
    T<:AbstractTargetSubject,
    S<:AbstractMCMCState,
    R<:AbstractRNG
}
    algorithm::A
    target::T
    state::S
    rng::R
    id::Int
    cycle::Int
    tuned::Bool
    converged::Bool
end

export MCMCIterator


nparams(chain::MCMCIterator) = nparams(chain.target)

sample_available(chain::MCMCIterator, status::Val = Val(:complete)) = sample_available(chain.state, status)

current_sample(chain::MCMCIterator, status::Val = Val(:complete)) = current_sample(chain.state, status)

current_sampleno(chain::MCMCIterator) = current_sampleno(chain.state)


function DensitySampleVector(chain::MCMCIterator)
    P = eltype(chain.state.current_sample.params)
    T = typeof(chain.state.current_sample.log_value)
    W = typeof(chain.state.current_sample.weight)
    m = size(chain.state.current_sample.params, 1)
    DensitySampleVector{P,T,W}(m)
end

function Base.push!(xs::DensitySampleVector, chain::MCMCIterator)
    push!(xs, chain.state)
    chain
end


function mcmc_step! end
export mcmc_step!


function mcmc_iterate! end
export mcmc_iterate!

function mcmc_iterate!(
    callback,
    chain::MCMCIterator,
    exec_context::ExecContext = ExecContext();
    max_nsamples::Int64 = Int64(1),
    max_nsteps::Int = 1000,
    max_time::Float64 = Inf,
    ll::LogLevel = LOG_NONE
)
    @log_msg ll "Starting iteration over MCMC chain $(chain.id)"

    cbfunc = Base.convert(AbstractMCMCCallback, callback)

    start_time = time()
    start_nsteps = nsteps(chain.state)
    start_nsamples = nsamples(chain.state)

    while (
        (chain.state.nsamples - start_nsamples) < max_nsamples &&
        (chain.state.nsteps - start_nsteps) < max_nsteps &&
        (time() - start_time) < max_time
    )
        mcmc_step!(cbfunc, chain, exec_context, ll + 1)
    end
    chain
end


exec_capabilities(mcmc_iterate!, f, chain::MCMCIterator) =
    exec_capabilities(target_logval, chain.target.tdensity, chain.state.proposed_sample.params)


function mcmc_iterate!(
    callbacks,
    chains::AbstractVector{<:MCMCIterator},
    exec_context::ExecContext = ExecContext();
    ll::LogLevel = LOG_NONE,
    kwargs...
)
    @log_msg ll "Starting iteration over $(length(chains)) MCMC chain(s)."

    cbv = mcmc_callback_vector(callbacks, chains)

    target_caps = exec_capabilities.(mcmc_iterate!, cbv, chains)
    (self_context, target_contexts) = negotiate_exec_context(exec_context, target_caps)

    threadsel = self_context.use_threads ? threads_all() : threads_this()
    idxs = eachindex(cbv, chains, target_contexts)
    onthreads(threadsel) do
        for i in workpartition(idxs, length(threadsel), threadid())
            mcmc_iterate!(cbv[i], chains[i], target_contexts[i]; ll=ll+1, kwargs...)
        end
    end
    chains
end


struct MCMCSpec{
    A<:MCMCAlgorithm,
    T<:AbstractTargetSubject,
    R<:AbstractRNGSeed
}
    algorithm::A
    target::T
    rngseed::R
end

export MCMCSpec

MCMCSpec(
    algorithm::MCMCAlgorithm,
    target::AbstractTargetSubject,
) = MCMCSpec(algorithm, target, Philox4xSeed())

MCMCSpec(
    algorithm::MCMCAlgorithm,
    tdensity::AbstractTargetDensity,
    bounds::AbstractParamBounds,
    rngseed::AbstractRNGSeed = Philox4xSeed()
) = MCMCSpec(algorithm, TargetSubject(tdensity, bounds), rngseed)


#=
# ToDo:

MCMCSpec(
    algorithm::MCMCAlgorithm,
    tdensity::AbstractTargetDensity,
    prior::XXX,
    bounds::AbstractParamBounds,
    rngseed::AbstractRNGSeed = Philox4xSeed()
) = MCMCSpec(algorithm, TargetSubject(tdensity, bounds), rngseed)
=#


function (spec::MCMCSpec)(
    id::Integer,
    exec_context::ExecContext = ExecContext()
)
    P = float(eltype(spec.target.bounds))
    m = nparams(spec.target)
    rng = spec.rngseed()
    MCMCIterator(
        spec.algorithm,
        spec.target,
        id,
        rng,
        Vector{P}(),
        exec_context,
    )
end


function (spec::MCMCSpec)(
    ids::AbstractVector,
    exec_context::ExecContext = ExecContext()
)
    spec.(ids, exec_context)
end


"""
    AbstractMCMCCallback <: Function

Subtypes (here, `X`) must support

    (::X)(level::Integer, chain::MCMCIterator) => nothing
    (::X)(level::Integer, tuner::AbstractMCMCTuner) => nothing

to be compabtible with `mcmc_iterate!`, `mcmc_tune_burnin!`, etc.
"""
abstract type AbstractMCMCCallback <: Function end
export AbstractMCMCCallback


@inline Base.convert(::Type{AbstractMCMCCallback}, x::AbstractMCMCCallback) = x

@inline Base.convert(::Type{Vector{AbstractMCMCCallback}}, V::Vector{<:AbstractMCMCCallback}) = V

Base.convert(::Type{Vector{AbstractMCMCCallback}}, V::Vector) =
    [convert(AbstractMCMCCallback, x) for x in V]


function mcmc_callback_vector(x, chains::AbstractVector{<:MCMCIterator})
    V = convert(Vector{AbstractMCMCCallback}, x)
    if eachindex(V) != eachindex(chains)
        throw(DimensionMismatch("Indices of callback vector incompatible with number of MCMC chains"))
    end
    V
end


mcmc_callback_vector(x::Tuple{}, chains::AbstractVector{<:MCMCIterator}) =
    [MCMCNopCallback() for _ in chains]



"""
    MCMCCallbackWrapper{F} <: AbstractMCMCCallback

Wraps a callable object to turn it into an `AbstractMCMCCallback`.

Constructor:

    MCMCCallbackWrapper(f::Any)

`f` needs to support the call syntax of an `AbstractMCMCCallback`.
"""
struct MCMCCallbackWrapper{F} <: AbstractMCMCCallback
    f::F
end


@inline (wrapper::MCMCCallbackWrapper)(args...) = wrapper.f(args...)

Base.convert(::Type{AbstractMCMCCallback}, f::Function) = MCMCCallbackWrapper(f)



struct MCMCNopCallback <: AbstractMCMCCallback end

(cb::MCMCNopCallback)(level::Integer, obj::Any) = nothing

Base.convert(::Type{AbstractMCMCCallback}, ::Tuple{}) = MCMCNopCallback()



struct MCMCMultiCallback{N,TP<:NTuple{N,AbstractMCMCCallback}} <: AbstractMCMCCallback
    funcs::TP

    MCMCMultiCallback(fs::NTuple{N,AbstractMCMCCallback}) where {N} =
        new{N, typeof(fs)}(fs)

    function fMCMCMultiCallback(fs::NTuple{N, Any}) where {N}
        fs_conv = map(x -> convert(AbstractMCMCCallback, x), fs)
        new{N, typeof(fs_conv)}(fs_conv)
    end
end


function (cb::MCMCMultiCallback)(level::Integer, obj::Any)
    for f in cb.funcs
        f(level, obj)
    end
    nothing
end

Base.convert(::Type{AbstractMCMCCallback}, fs::Tuple) = MCMCMultiCallback(fs)



struct MCMCPushCallback{T} <: AbstractMCMCCallback
    max_level::Int
    target::T
end

export MCMCPushCallback

MCMCPushCallback(target) = MCMCPushCallback(1, target)

MCMCPushCallback(max_level::Int, t, ts...) =
    MCMCMultiCallback(map(x -> MCMCPushCallback(max_level, x), (t, ts...)))


function (cb::MCMCPushCallback)(level::Integer, chain::MCMCIterator)
    if (level <= cb.max_level)
        push!(cb.target, chain)
    end
    nothing
end

(cb::MCMCPushCallback)(level::Integer, obj::Any) = nothing


Base.convert(::Type{AbstractMCMCCallback}, x::BATDataVector) = MCMCPushCallback(x)
