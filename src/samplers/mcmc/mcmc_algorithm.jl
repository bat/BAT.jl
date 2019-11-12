# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    abstract type MCMCAlgorithm <: AbstractSamplingAlgorithm end

!!! note

    The details of the `MCMCIterator` and `MCMCAlgorithm` API (see below)
    currently do not form part of the BAT-API, and may change without notice.

The following methods must be defined for subtypes (e.g.
for `SomeAlgorithm<:MCMCAlgorithm`):

```julia
(spec::MCMCSpec{<:SomeAlgorithm})(chainid::Integer)::MCMCIterator
```

In some cases, these custom methods may be necessary (default methods are
defined for `MCMCAlgorithm`):

```julia
BAT.initial_params!(
    params::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}},
    rng::AbstractRNG,
    posterior::AbstractPosteriorDensity,
    algorithm::SomeAlgorithm
)
```

To implement a new MCMC algorithm, subtypes of both `MCMCAlgorithm` and
[`MCMCIterator`](@ref) are required.
"""
abstract type MCMCAlgorithm <: AbstractSamplingAlgorithm end
export MCMCAlgorithm


"""
    BAT.initial_params!(
        params::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}},
        rng::AbstractRNG,
        posterior::AbstractPosteriorDensity,
        algorithm::MCMCAlgorithm
    )::typeof(params)

*BAT-internal, not part of stable public API.*

Fill `params` with random initial parameters suitable for `posterior` and
`algorithm`. The default implementation will try to draw the initial
parameters from the prior of the posterior.
"""
function initial_params! end

initial_params!(
    params::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}},
    rng::AbstractRNG,
    posterior::AbstractPosteriorDensity,
    algorithm::MCMCAlgorithm
) = rand!(rng, sampler(getprior(posterior)), params)


@doc doc"""
    MCMCSpec{
        A<:MCMCAlgorithm,
        M<:AbstractPosteriorDensity,
        R<:AbstractRNGSeed
    }

*BAT-internal, not part of stable public API.*

Specifies a Bayesian MCMC chain.

Constructor:

```julia
MCMCSpec(
    algorithm::MCMCAlgorithm,
    posterior::AbstractPosteriorDensity,
    rngseed::AbstractRNGSeed = AbstractRNGSeed()
)
```

Markov-chain instances, represented by objects of type [`MCMCIterator`](@ref),
are be created via

```julia
(spec::MCMCSpec)(chainid::Integer)
```
"""
struct MCMCSpec{
    A<:MCMCAlgorithm,
    M<:AbstractPosteriorDensity,
    R<:AbstractRNGSeed
}
    algorithm::A
    posterior::M
    rngseed::R
end


# TODO/Decision: Make MCMCSpec a subtype of StatsBase.Sampleable{Multivariate,Continuous}?


MCMCSpec(
    algorithm::MCMCAlgorithm,
    posterior::AbstractPosteriorDensity
) = MCMCSpec(algorithm, posterior, AbstractRNGSeed())

@deprecate MCMCSpec(
    algorithm::MCMCAlgorithm,
    likelihood::Union{AbstractDensity,Distribution{Multivariate,Continuous}},
    prior::Union{AbstractDensity,ParamVolumeBounds},
    rngseed::AbstractRNGSeed = AbstractRNGSeed()
) MCMCSpec(algorithm, PosteriorDensity(likelihood, prior), rngseed)




@with_kw struct MCMCIteratorInfo
    id::Int64
    cycle::Int
    tuned::Bool
    converged::Bool
end



@doc doc"""
    abstract type MCMCIterator end

Represents the current state of a MCMC chain.

!!! note

    The details of the `MCMCIterator` and `MCMCAlgorithm` API (see below)
    currently do not form part of the BAT-API, and may change without notice.

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

BAT.current_sample(chain::SomeMCMCIter)::DensitySample

BAT.sample_type(chain::SomeMCMCIter)::Type{<:DensitySample}

BAT.samples_available(chain::SomeMCMCIter, nonzero_weights::Bool = false)::Bool

BAT.get_samples!(appendable, chain::SomeMCMCIter, nonzero_weights::Bool)::typeof(appendable)

BAT.next_cycle!(chain::SomeMCMCIter)::SomeMCMCIter

BAT.mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::SomeMCMCIter
)
```

The following methods are implemented by default:

```julia
algorithm(chain::MCMCIterator)
getposterior(chain::MCMCIterator)
rngseed(chain::MCMCIterator)
nparams(chain::MCMCIterator)
DensitySampleVector(chain::MCMCIterator)
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

function current_sample end

function sample_type end

function samples_available end

function get_samples! end

function next_cycle! end

function mcmc_step! end


algorithm(chain::MCMCIterator) = mcmc_spec(chain).algorithm

getposterior(chain::MCMCIterator) = mcmc_spec(chain).posterior

rngseed(chain::MCMCIterator) = mcmc_spec(chain).rngseed

nparams(chain::MCMCIterator) = nparams(getposterior(chain))

DensitySampleVector(chain::MCMCIterator) = DensitySampleVector(sample_type(chain), nparams(chain))



function mcmc_iterate! end


function mcmc_iterate!(
    callback,
    chain::MCMCIterator;
    max_nsamples::Int64 = Int64(1),
    max_nsteps::Int64 = Int64(1000),
    max_time::Float64 = Inf
)
    @debug "Starting iteration over MCMC chain $(chain.info.id), max_nsamples = $max_nsamples, max_nsteps = $max_nsteps, max_time = $max_time"

    cbfunc = Base.convert(AbstractMCMCCallback, callback)

    start_time = time()
    start_nsteps = nsteps(chain)
    start_nsamples = nsamples(chain)

    while (
        (nsamples(chain) - start_nsamples) < max_nsamples &&
        (nsteps(chain) - start_nsteps) < max_nsteps &&
        (time() - start_time) < max_time
    )
        mcmc_step!(cbfunc, chain)
    end

    end_time = time()
    elapsed_time = end_time - start_time

    @debug "Finished iteration over MCMC chain $(chain.info.id), nsamples = $(nsamples(chain)), nsteps = $(nsteps(chain)), time = $(Float32(elapsed_time))"

    chain
end


function mcmc_iterate!(
    callbacks,
    chains::AbstractVector{<:MCMCIterator};
    kwargs...
)
    if isempty(chains)
        @debug "No MCMC chain(s) to iterate over."
        return chains
    else
        @debug "Starting iteration over $(length(chains)) MCMC chain(s)"
    end

    cbv = mcmc_callback_vector(callbacks, eachindex(chains))

    idxs = eachindex(cbv, chains)
    @sync for i in idxs
        @mt_async mcmc_iterate!(cbv[i], chains[i]; kwargs...)
    end

    chains
end
