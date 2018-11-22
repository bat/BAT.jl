# This file is a part of BAT.jl, licensed under the MIT License (MIT).


doc"""
    abstract type MCMCAlgorithm end

The following methods must be defined for subtypes (e.g.
for `SomeAlgorithm<:MCMCAlgorithm`):

```julia
(spec::MCMCSpec{<:SomeAlgorithm})(
    chainid::Integer,
    exec_context::ExecContext
)::MCMCIterator
```

In some cases, these custom methods may necessary (default methods are defined
for `MCMCAlgorithm`):

```julia
BAT.initial_params!(
    params::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}},
    rng::AbstractRNG,
    model::AbstractBayesianModel,
    algorithm::SomeAlgorithm
)
```

To implement a new MCMC algorithm, subtypes of both `MCMCAlgorithm` and
[`MCMCIterator`](@ref) are required.
"""
abstract type MCMCAlgorithm end
export MCMCAlgorithm


"""
    BAT.initial_params!(
        params::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}},
        rng::AbstractRNG,
        model::AbstractBayesianModel,
        algorithm::MCMCAlgorithm
    )::typeof(params)

Fill `params` with random initial parameters suitable for `model` and
`algorithm`. The default implementation will try to draw the initial
parameters from the prior of the model.
"""
function initial_params! end

initial_params!(
    params::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}},
    rng::AbstractRNG,
    model::AbstractBayesianModel,
    algorithm::MCMCAlgorithm
) = rand!(rng, prior(model), params)


doc"""
    MCMCSpec{
        A<:MCMCAlgorithm,
        M<:AbstractBayesianModel,
        R<:AbstractRNGSeed
    }

Specifies a Bayesian MCMC chain.

Constructor:

```julia
MCMCSpec(
    algorithm::MCMCAlgorithm,
    model::AbstractBayesianModel,
    rngseed::AbstractRNGSeed = AbstractRNGSeed()
)
```

Markov-chain instances, represented by objects of type [`MCMCIterator`](@ref),
are be created via

```julia
(spec::MCMCSpec)(chainid::Integer)
(spec::MCMCSpec)(chainid::Integer, exec_context::ExecContext)
```
"""
struct MCMCSpec{
    A<:MCMCAlgorithm,
    M<:AbstractBayesianModel,
    R<:AbstractRNGSeed
}
    algorithm::A
    model::M
    rngseed::R
end

export MCMCSpec


# TODO/Decision: Make MCMCSpec a subtype of StatsBase.Sampleable{Multivariate,Continuous}?


MCMCSpec(
    algorithm::MCMCAlgorithm,
    model::AbstractBayesianModel
) = MCMCSpec(algorithm, model, AbstractRNGSeed())

@deprecate MCMCSpec(
    algorithm::MCMCAlgorithm,
    likelihood::Union{AbstractDensity,Distribution{Multivariate,Continuous}},
    prior::Union{AbstractDensity,ParamVolumeBounds},
    rngseed::AbstractRNGSeed = AbstractRNGSeed()
) MCMCSpec(algorithm, BayesianModel(likelihood, prior), rngseed)


(spec::MCMCSpec)(chainid::Integer) = spec(chainid, ExecContext())




@with_kw struct MCMCIteratorInfo
    id::Int64
    cycle::Int
    tuned::Bool
    converged::Bool
end



doc"""
    abstract type MCMCIterator end

Represents the current state of a MCMC chain.

To implement a new MCMC algorithm, subtypes of both [`MCMCAlgorithm`](@ref)
and `MCMCIterator` are required.

The following methods must be defined for subtypes of `MCMCIterator` (e.g.
`SomeMCMCIter<:MCMCIterator`):

```julia
BAT.mcmc_spec(chain::SomeMCMCIter)::MCMCSpec

BAT.getrng(chain::SomeMCMCIter)::AbstractRNG

BAT.mcmc_info(chain::SomeMCMCIter)::MCMCIteratorInfo

BAT.nsteps(chain::SomeMCMCIter)::Int

BAT.nsamples(chain::SomeMCMCIter)::Int

BAT.sample_type(chain::SomeMCMCIter)::Type{<:DensitySample}

BAT.samples_available(chain::SomeMCMCIter, nonzero_weights::Bool = false)::Bool

BAT.get_samples!(appendable, chain::SomeMCMCIter, nonzero_weights::Bool)::typeof(appendable)

BAT.get_sample_ids!(appendable, chain::SomeMCMCIter, nonzero_weights::Bool)::typeof(appendable)

BAT.next_cycle!(chain::SomeMCMCIter)::SomeMCMCIter

BAT.exec_capabilities(mcmc_step!, callback::AbstractMCMCCallback, chain::SomeMCMCIter)

BAT.mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::SomeMCMCIter,
    exec_context::ExecContext,
    ll::LogLevel
)
```

The following methods are implemented by default:

```julia
algorithm(chain::MCMCIterator)
getmodel(chain::MCMCIterator)
rngseed(chain::MCMCIterator)
nparams(chain::MCMCIterator)
DensitySampleVector(chain::MCMCIterator)
MCMCSampleIDVector(chain::MCMCIterator)
mcmc_iterate!(callback, chain::MCMCIterator, ...)
mcmc_iterate!(callbacks, chains::AbstractVector{<:MCMCIterator}, ...)
```
"""
abstract type MCMCIterator end
export MCMCIterator


function mcmc_spec end

function getrng end

function mcmc_info end

function nsteps end

function nsamples end

function sample_type end

function samples_available end

function get_samples! end

function get_sample_ids! end

function next_cycle! end

function mcmc_step! end


algorithm(chain::MCMCIterator) = mcmc_spec(chain).algorithm

getmodel(chain::MCMCIterator) = mcmc_spec(chain).model

rngseed(chain::MCMCIterator) = mcmc_spec(chain).rngseed

nparams(chain::MCMCIterator) = nparams(likelihood(getmodel(chain)))

DensitySampleVector(chain::MCMCIterator) = DensitySampleVector(sample_type(chain), nparams(chain))

MCMCSampleIDVector(chain::MCMCIterator) = MCMCSampleIDVector()



function mcmc_iterate! end


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
    @log_msg ll "Starting iteration over MCMC chain $(chain.info.id)"

    cbfunc = Base.convert(AbstractMCMCCallback, callback)

    start_time = time()
    start_nsteps = nsteps(chain)
    start_nsamples = nsamples(chain)

    while (
        (nsamples(chain) - start_nsamples) < max_nsamples &&
        (nsteps(chain) - start_nsteps) < max_nsteps &&
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

    cbv = mcmc_callback_vector(callbacks, eachindex(chains))

    target_caps = exec_capabilities.(mcmc_iterate!, cbv, chains)
    (self_context, target_contexts) = negotiate_exec_context(exec_context, target_caps)

    threadsel = self_context.use_threads ? allthreads() : (threadid():threadid())
    idxs = eachindex(cbv, chains, target_contexts)

    @onthreads threadsel begin
        for i in workpart(idxs, threadsel, threadid())
            mcmc_iterate!(cbv[i], chains[i], target_contexts[i]; ll=ll+1, kwargs...)
        end
    end

    chains
end
