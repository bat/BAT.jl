# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    abstract type MCMCAlgorithm end

!!! note

    The details of the `MCMCIterator` and `MCMCAlgorithm` API (see below)
    currently do not form part of the BAT-API, and may change without notice.

The following methods must be defined for subtypes (e.g.
for `SomeAlgorithm<:MCMCAlgorithm`):

```julia
MCMCIterator(
    rng::AbstractRNG,
    algorithm::SomeAlgorithm,
    density::AbstractDensity,
    chainid::Int,
    startpos::AbstractVector{<:Real}
)
```

To implement a new MCMC algorithm, subtypes of both `MCMCAlgorithm` and
[`MCMCIterator`](@ref) are required.
"""
abstract type MCMCAlgorithm end
export MCMCAlgorithm


"""
    BAT.mcmc_startval!(
        x::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}},
        rng::AbstractRNG,
        posterior::AbstractPosteriorDensity,
        algorithm::MCMCAlgorithm
    )::typeof(x)

*BAT-internal, not part of stable public API.*

Fill `x` a random initial argument suitable for `posterior` and
`algorithm`. The default implementation will try to draw the initial
argument value from the prior of the posterior.
"""
function mcmc_startval! end

mcmc_startval!(
    x::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}},
    rng::AbstractRNG,
    posterior::AbstractPosteriorDensity,
    algorithm::MCMCAlgorithm
) = rand!(rng, sampler(getprior(posterior)), x)


@with_kw struct MCMCIteratorInfo
    id::Int64
    cycle::Int
    tuned::Bool
    converged::Bool
end

struct MCMCSampleGenerator{T<:MCMCIterator} <: AbstractSampleGenerator
    _chains::T
end

getalgorithm(sg::MCMCSampleGenerator) = sg._chains[1].spec.algorithm


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

BAT.get_samples!(samples::DensitySampleVector, chain::SomeMCMCIter, nonzero_weights::Bool)::typeof(samples)

BAT.next_cycle!(chain::SomeMCMCIter)::SomeMCMCIter

BAT.mcmc_step!(
    chain::SomeMCMCIter
    callback::Function,
)::nothing
```

The following methods are implemented by default:

```julia
algorithm(chain::MCMCIterator)
getposterior(chain::MCMCIterator)
rngseed(chain::MCMCIterator)
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

DensitySampleVector(chain::MCMCIterator) = DensitySampleVector(sample_type(chain), totalndof(getposterior(chain)))



function mcmc_iterate! end


function mcmc_iterate!(
    output::OptionalDensitySampleVector,
    chain::MCMCIterator;
    max_nsamples::Int64 = Int64(1),
    max_nsteps::Int64 = Int64(1000),
    max_time::Float64 = Inf,
    nonzero_weights::Bool = true,
    callback::Function = noop_func
)
    @debug "Starting iteration over MCMC chain $(chain.info.id), max_nsamples = $max_nsamples, max_nsteps = $max_nsteps, max_time = $max_time"

    start_time = time()
    start_nsteps = nsteps(chain)
    start_nsamples = nsamples(chain)

    while (
        (nsamples(chain) - start_nsamples) < max_nsamples &&
        (nsteps(chain) - start_nsteps) < max_nsteps &&
        (time() - start_time) < max_time
    )
        mcmc_step!(chain, callback)
        if !isnothing(output)
            get_samples!(output, chain, nonzero_weights)
        end
    end

    end_time = time()
    elapsed_time = end_time - start_time

    @debug "Finished iteration over MCMC chain $(chain.info.id), nsamples = $(nsamples(chain)), nsteps = $(nsteps(chain)), time = $(Float32(elapsed_time))"

    nothing
end


function mcmc_iterate!(
    output::OptionalDensitySampleVector,
    chains::AbstractVector{<:MCMCIterator};
    kwargs...
)
    if isempty(chains)
        @debug "No MCMC chain(s) to iterate over."
        return chains
    else
        @debug "Starting iteration over $(length(chains)) MCMC chain(s)"
    end

    chain_outputs = if isnothing(output)
        map(x -> nothing, chains)
    else
        map(x -> similar(output, 0), chains)
    end

    idxs = eachindex(chain_outputs, chains)
    @sync for i in idxs
        @mt_async mcmc_iterate!(chain_outputs[i], chains[i]; kwargs...)
    end

    if !isnothing(output)
        for (o in chain_outputs)
            append!(output, o)
        end
    end

    nothing
end
