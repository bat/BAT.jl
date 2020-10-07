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
    chainid::Integer,
    [startpos::AbstractVector{<:Real}]
)
```

To implement a new MCMC algorithm, subtypes of both `MCMCAlgorithm` and
[`MCMCIterator`](@ref) are required.
"""
abstract type MCMCAlgorithm end
export MCMCAlgorithm


function get_mcmc_tuning end


@doc doc"""
abstract type MCMCInitAlgorithm end

Abstract type for MCMC initialization algorithms.
"""
abstract type MCMCInitAlgorithm end
export MCMCInitAlgorithm



"""
MCMCTuningAlgorithm

Abstract super-type for MCMC tuning algorithms.
"""
abstract type MCMCTuningAlgorithm end
export MCMCTuningAlgorithm



"""
MCMCBurninAlgorithm

Abstract super-type for MCMC burn-in algorithms.
"""
abstract type MCMCBurninAlgorithm end
export MCMCBurninAlgorithm


"""
MCMCBurninAlgorithm

Abstract super-type for MCMC convergence tests.
"""
abstract type MCMCConvergenceTest end
export MCMCConvergenceTest



@with_kw struct MCMCIteratorInfo
    id::Int32
    cycle::Int32
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
BAT.getalgorithm(chain::SomeMCMCIter)::MCMCAlgorithm

BAT.getdensity(chain::SomeMCMCIter)::AbstractDensity

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
getalgorithm(chain::MCMCIterator)
getdensity(chain::MCMCIterator)
DensitySampleVector(chain::MCMCIterator)
mcmc_iterate!(chain::MCMCIterator, ...)
mcmc_iterate!(chains::AbstractVector{<:MCMCIterator}, ...)
isvalidchain(chain::MCMCIterator)
isviablechain(chain::MCMCIterator)
```
"""
abstract type MCMCIterator end
export MCMCIterator


function getalgorithm end

function getdensity end

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



DensitySampleVector(chain::MCMCIterator) = DensitySampleVector(sample_type(chain), totalndof(getdensity(chain)))



function mcmc_iterate! end


function mcmc_iterate!(
    output::Union{DensitySampleVector,Nothing},
    chain::MCMCIterator;
    max_nsamples::Integer = 1,
    max_nsteps::Integer = 1000,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func
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
        mcmc_step!(chain)
        callback(Val(:mcmc_step), chain)
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
    outputs::Union{AbstractVector{<:DensitySampleVector},Nothing},
    chains::AbstractVector{<:MCMCIterator};
    kwargs...
)
    if isempty(chains)
        @debug "No MCMC chain(s) to iterate over."
        return chains
    else
        @debug "Starting iteration over $(length(chains)) MCMC chain(s)"
    end

    if !isnothing(outputs)
        #@sync
        for i in eachindex(outputs, chains)
            #@mt_async
            mcmc_iterate!(outputs[i], chains[i]; kwargs...)
        end
    else
        #@sync
        for i in eachindex(chains)
            #@mt_async 
            mcmc_iterate!(nothing, chains[i]; kwargs...)
        end
    end

    nothing
end


isvalidchain(chain::MCMCIterator) = current_sample(chain).logd > -Inf

isviablechain(chain::MCMCIterator) = nsamples(chain) >= 2



"""
    BAT.MCMCSampleGenerator

*BAT-internal, not part of stable public API.*

MCMC sample generator.

Constructors:

```julia
MCMCSampleGenerator(chain::AbstractVector{<:MCMCIterator})
```
"""
struct MCMCSampleGenerator{T<:AbstractVector{<:MCMCIterator}} <: AbstractSampleGenerator
    chains::T
end

getalgorithm(sg::MCMCSampleGenerator) = sg.chains[1].spec.algorithm



abstract type AbstractMCMCTunerInstance end

function tuning_init! end

function mcmc_init! end

function mcmc_burnin! end


function isvalidchain end

function isviablechain end
