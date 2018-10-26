# This file is a part of BAT.jl, licensed under the MIT License (MIT).


doc"""
    abstract type AbstractMCMCState end

The following methods must be defined for subtypes (e.g.
for `SomeMCMCState<:AbstractMCMCState` and ):

```julia
nparams(state::SomeMCMCState)

nsteps(state::SomeMCMCState)

nsamples(state::SomeMCMCState)

next_cycle!(state::SomeMCMCState)

density_sample_type(state::SomeMCMCState)
```
"""
abstract type AbstractMCMCState end

function density_sample_type end



doc"""
    abstract type MCMCAlgorithm{S<:AbstractMCMCState} <: BATAlgorithm end

The following methods must be defined for subtypes (e.g.
for `SomeAlgorithm<:MCMCAlgorithm`):

```julia

sample_weight_type(::Type{SomeAlgorithm}) where {Q,W,WS} = W

MCMCIterator(
    algorithm::SomeAlgorithm,
    likelihood::AbstractDensity,
    prior::AbstractDensity,
    id::Int64,
    rng::AbstractRNG,
    initial_params::AbstractVector{P} = default_value,
    exec_context::ExecContext = ExecContext(),
)

exec_capabilities(mcmc_step!, callback::AbstractMCMCCallback, chain::MCMCIterator{<:SomeAlgorithm})

mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::MCMCIterator{<:SomeAlgorithm},
    exec_context::ExecContext,
    ll::LogLevel
)

nsamples_available(chain::MCMCIterator{<:SomeAlgorithm}, nonzero_weights::Bool = false)::Integer

get_samples!(appendable, chain::MCMCIterator{<:SomeAlgorithm}, nonzero_weights::Bool)::typeof(appendable)

get_sample_ids!(appendable, chain::MCMCIterator{<:SomeAlgorithm}, nonzero_weights::Bool)::typeof(appendable)
```

In some cases, these custom methods may necessary (default methods are defined
for `MCMCAlgorithm`):

```julia
rand_initial_params!(rng::AbstractRNG, algorithm::SomeAlgorithm, prior::AbstractDensity, x::StridedVecOrMat{<:Real})
```
"""
abstract type MCMCAlgorithm{S<:AbstractMCMCState} <: BATAlgorithm end
export MCMCAlgorithm


function sample_weight_type end


rand_initial_params!(rng::AbstractRNG, algorithm::MCMCAlgorithm, prior::AbstractDensity, x::StridedVecOrMat{<:Real}) =
    rand!(rng, prior, x)

# ToDo:
# function rand_initial_params!(rng::AbstractRNG, algorithm::MCMCAlgorithm, prior::TransformedDensity, x::StridedVecOrMat{<:Real}) =
#     rand_initial_params!(rng, algorithm, parent(prior), x)
#     ... apply transformation to x ...
#     x
# end



mutable struct MCMCIterator{
    A<:MCMCAlgorithm,
    T<:AbstractDensity,
    S<:AbstractMCMCState,
    R<:AbstractRNG
}
    algorithm::A
    target::T
    state::S
    rng::R
    id::Int64
    cycle::Int
    tuned::Bool
    converged::Bool
end

export MCMCIterator


nparams(chain::MCMCIterator) = nparams(chain.target)


DensitySampleVector(chain::MCMCIterator) = DensitySampleVector(density_sample_type(chain.state), nparams(chain.state))


reset_rng_counters!(chain::MCMCIterator) =
    reset_rng_counters!(chain.rng, chain.id, chain.cycle, nsteps(chain.state))


function next_cycle!(chain::MCMCIterator)
    next_cycle!(chain.state)
    chain.cycle += 1
    reset_rng_counters!(chain)
    chain
end


function mcmc_step! end
export mcmc_step!


function mcmc_iterate! end
export mcmc_iterate!


exec_capabilities(mcmc_iterate!, callback, chain::MCMCIterator) =
    exec_capabilities(mcmc_step!, Base.convert(AbstractMCMCCallback, callback), chain)

function mcmc_iterate!(
    callback,
    chain::MCMCIterator,
    exec_context::ExecContext = ExecContext();
    max_nsamples::Int64 = Int64(1),
    max_nsteps::Int64 = Int64(1000),
    max_time::Float64 = Inf,
    ll::LogLevel = LOG_NONE
)
    @log_msg ll "Starting iteration over MCMC chain $(chain.id)"

    cbfunc = Base.convert(AbstractMCMCCallback, callback)

    start_time = time()
    start_nsteps = nsteps(chain.state)
    start_nsamples = nsamples(chain.state)

    while (
        (nsamples(chain.state) - start_nsamples) < max_nsamples &&
        (nsteps(chain.state) - start_nsteps) < max_nsteps &&
        (time() - start_time) < max_time
    )
        mcmc_step!(cbfunc, chain, exec_context, ll + 1)
    end
    chain
end


function mcmc_iterate!(
    callbacks,
    chains::AbstractVector{<:MCMCIterator},
    exec_context::ExecContext = ExecContext();
    ll::LogLevel = LOG_NONE,
    kwargs...
)
    if isempty(chains)
        @log_msg ll "No MCMC chain(s) to iterate over."
        return chains
    else
        @log_msg ll "Starting iteration over $(length(chains)) MCMC chain(s)."
    end

    cbv = mcmc_callback_vector(callbacks, chains)

    target_caps = exec_capabilities.(mcmc_iterate!, cbv, chains)
    (self_context, target_contexts) = negotiate_exec_context(exec_context, target_caps)

    threadsel = self_context.use_threads ? allthreads() : (threadid():threadid())
    idxs = eachindex(cbv, chains, target_contexts)

    @onthreads threadsel begin
        for i in workpartition(idxs, length(threadsel), threadid())
            mcmc_iterate!(cbv[i], chains[i], target_contexts[i]; ll=ll+1, kwargs...)
        end
    end

    chains
end



# TODO/Decision: Make MCMCSpec a subtype of Sampleable{Multivariate,Continuous}?

struct MCMCSpec{
    A<:MCMCAlgorithm,
    L<:AbstractDensity,
    P<:AbstractDensity,
    R<:AbstractRNGSeed
}
    algorithm::A
    likelihood::L
    prior::P
    rngseed::R
end

export MCMCSpec

MCMCSpec(
    algorithm::MCMCAlgorithm,
    likelihood::AbstractDensity,
    prior::Union{AbstractDensity,ParamVolumeBounds},
    rngseed::AbstractRNGSeed = AbstractRNGSeed()
) = MCMCSpec(algorithm, likelihood, convert(AbstractDensity, prior), AbstractRNGSeed())

MCMCSpec(
    algorithm::MCMCAlgorithm,
    distribution::Distribution{Multivariate,Continuous},
    prior::Union{AbstractDensity,ParamVolumeBounds},
    rngseed::AbstractRNGSeed = AbstractRNGSeed()
) = MCMCSpec(algorithm, MvDistDensity(distribution), convert(AbstractDensity, prior), rngseed)


function (spec::MCMCSpec)(
    id::Int64,
    exec_context::ExecContext = ExecContext()
)
    P = float(eltype(param_bounds(spec.prior)))
    rng = spec.rngseed()

    MCMCIterator(
        spec.algorithm,
        spec.likelihood,
        spec.prior,
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



@doc """
    AbstractMCMCCallback <: Function

Subtypes (e.g. `X`) must support

    (::X)(level::Integer, chain::MCMCIterator) => nothing
    (::X)(level::Integer, tuner::AbstractMCMCTuner) => nothing

to be compabtible with `mcmc_iterate!`, `mcmc_tune_burnin!`, etc.
"""
abstract type AbstractMCMCCallback <: Function end
export AbstractMCMCCallback

@inline Base.convert(::Type{AbstractMCMCCallback}, x::AbstractMCMCCallback) = x

@inline Base.convert(::Type{Vector{<:AbstractMCMCCallback}}, V::Vector{<:AbstractMCMCCallback}) = V

Base.convert(::Type{Vector{<:AbstractMCMCCallback}}, V::Vector) =
            [convert(AbstractMCMCCallback, x) for x in V]


function mcmc_callback_vector(x, chains::AbstractVector{<:MCMCIterator})
    V = convert(Vector{<:AbstractMCMCCallback}, x)
    if eachindex(V) != eachindex(chains)
        throw(DimensionMismatch("Indices of callback vector incompatible with number of MCMC chains"))
    end
    V
end

mcmc_callback_vector(x::Tuple{}, chains::AbstractVector{<:MCMCIterator}) =
    [MCMCNopCallback() for _ in chains]



@doc """
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

    MCMCMultiCallback(fs::Vararg{AbstractMCMCCallback,N}) where {N} =
        new{N, typeof(fs)}(fs)

    function fMCMCMultiCallback(fs::Vararg{Any,N}) where {N}
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

Base.convert(::Type{AbstractMCMCCallback}, fs::Tuple) = MCMCMultiCallback(fs...)



struct MCMCAppendCallback{T,F<:Function} <: AbstractMCMCCallback
    appendable::T
    max_level::Int
    get_data_func!::F
    nonzero_weights::Bool
end

export MCMCAppendCallback


function (cb::MCMCAppendCallback)(level::Integer, chain::MCMCIterator)
    if (level <= cb.max_level)
        cb.get_data_func!(cb.appendable, chain, cb.nonzero_weights)
    end
    nothing
end


Base.convert(::Type{AbstractMCMCCallback}, x::DensitySampleVector) = MCMCAppendCallback(x)

MCMCAppendCallback(x::DensitySampleVector, nonzero_weights::Bool = true) =
    MCMCAppendCallback(x, 1, get_samples!, nonzero_weights)
