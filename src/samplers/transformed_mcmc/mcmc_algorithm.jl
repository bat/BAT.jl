# This file is a part of BAT.jl, licensed under the MIT License (MIT).



function get_mcmc_tuning end #TODO: still needed


"""
    abstract type TransformedMCMCInitAlgorithm

Abstract type for MCMC initialization algorithms.
"""
abstract type TransformedMCMCInitAlgorithm end
export TransformedMCMCInitAlgorithm

#TODO AC: reactivate
#apply_trafo_to_init(trafo::Function, initalg::TransformedMCMCInitAlgorithm) = initalg



"""
    abstract type TransformedMCMCTuningAlgorithm

Abstract type for MCMC tuning algorithms.
"""
abstract type TransformedMCMCTuningAlgorithm end
export TransformedMCMCTuningAlgorithm



"""
    abstract type TransformedMCMCBurninAlgorithm

Abstract type for MCMC burn-in algorithms.
"""
abstract type TransformedMCMCBurninAlgorithm end
export TransformedMCMCBurninAlgorithm



@with_kw struct TransformedMCMCIteratorInfo
    id::Int32
    cycle::Int32
    tuned::Bool
    converged::Bool
end


# TODO AC: reactivate
# """
#     abstract type MCMCIterator end

# Represents the current state of an MCMC chain.

# !!! note

#     The details of the `MCMCIterator` and `MCMCAlgorithm` API (see below)
#     currently do not form part of the stable API and are subject to change
#     without deprecation.

# To implement a new MCMC algorithm, subtypes of both [`MCMCAlgorithm`](@ref)
# and `MCMCIterator` are required.

# The following methods must be defined for subtypes of `MCMCIterator` (e.g.
# `SomeMCMCIter<:MCMCIterator`):

# ```julia

# BAT.getmeasure(chain::SomeMCMCIter)::BATMeasure

# BAT.getcontext(chain::SomeMCMCIter)::BATContext

# BAT.mcmc_info(chain::SomeMCMCIter)::TransformedMCMCIteratorInfo

# BAT.nsteps(chain::SomeMCMCIter)::Int

# BAT.nsamples(chain::SomeMCMCIter)::Int

# BAT.current_sample(chain::SomeMCMCIter)::DensitySample

# BAT.sample_type(chain::SomeMCMCIter)::Type{<:DensitySample}

# BAT.samples_available(chain::SomeMCMCIter, nonzero_weights::Bool = false)::Bool

# BAT.get_samples!(samples::DensitySampleVector, chain::SomeMCMCIter, nonzero_weights::Bool)::typeof(samples)

# BAT.next_cycle!(chain::SomeMCMCIter)::SomeMCMCIter

# BAT.mcmc_step!(
#     chain::SomeMCMCIter
#     callback::Function,
# )::nothing
# ```

# The following methods are implemented by default:

# ```julia
# getalgorithm(chain::MCMCIterator)
# getmeasure(chain::MCMCIterator)
# DensitySampleVector(chain::MCMCIterator)
# mcmc_iterate!(chain::MCMCIterator, ...)
# mcmc_iterate!(chains::AbstractVector{<:MCMCIterator}, ...)
# isvalidchain(chain::MCMCIterator)
# isviablechain(chain::MCMCIterator)
# ```
# """
# abstract type MCMCIterator end
# export MCMCIterator


#TODO AC: reactivate
# function Base.show(io::IO, chain::MCMCIterator)
#     print(io, Base.typename(typeof(chain)).name, "(")
#     print(io, "id = "); show(io, mcmc_info(chain).id)
#     print(io, ", nsamples = "); show(io, nsamples(chain))
#     print(io, ", density = "); show(io, getmeasure(chain))
#     print(io, ")") 
# end


function getalgorithm end

function getmeasure end

function mcmc_info end

function nsteps end

function nsamples end

function current_sample end

function sample_type end

function samples_available end

function get_samples! end

function next_cycle! end

function mcmc_step! end


# TODO AC: reactivate
#DensitySampleVector(chain::MCMCIterator) = DensitySampleVector(sample_type(chain), totalndof(getmeasure(chain)))


abstract type TransformedAbstractMCMCTunerInstance end


function tuning_init! end

function tuning_postinit! end

function tuning_reinit! end

function tuning_update! end

function tuning_finalize! end

function tuning_callback end


function mcmc_init! end

function mcmc_burnin! end


function isvalidchain end

function isviablechain end



function mcmc_iterate! end

"""
    BAT.TransformedMCMCSampleGenerator

*BAT-internal, not part of stable public API.*

MCMC sample generator.

Constructors:

```julia
TransformedMCMCSampleGenerator(chain::AbstractVector{<:MCMCIterator})
```
"""
struct TransformedMCMCSampleGenerator{
    T<:AbstractVector{<:MCMCIterator},
    A<:AbstractSamplingAlgorithm,
} <: AbstractSampleGenerator
    chains::T
    algorithm::A
end

getalgorithm(sg::TransformedMCMCSampleGenerator) = sg.algorithm

function Base.show(io::IO, generator::TransformedMCMCSampleGenerator)
    if get(io, :compact, false)
        print(io, nameof(typeof(generator)), "(")
        if !isempty(generator.chains)
            show(io, first(generator.chains))
            print(io, ", …")
        end
        print(io, ")")
    else
        println(io, nameof(typeof(generator)), ":")
        chains = generator.chains
        nchains = length(chains)
        n_tuned_chains = count(c -> c.info.tuned, chains)
        n_converged_chains = count(c -> c.info.converged, chains)
        print(io, "algorithm: ")
        show(io, "text/plain", getalgorithm(generator))
        println(io)
        println(io, "number of chains:", repeat(' ', 12), nchains)
        println(io, "number of chains tuned:", repeat(' ', 6), n_tuned_chains)
        println(io, "number of chains converged:", repeat(' ', 2), n_converged_chains)
        println(io, "number of points…")
        println(io, repeat(' ',10), "… in 1th chain:", repeat(' ', 4), nsamples(first(chains)))
        print(io, repeat(' ',10), "… on average:", repeat(' ', 6), div(sum(nsamples.(chains)), nchains))
    end
end


function bat_report!(md::Markdown.MD, generator::TransformedMCMCSampleGenerator)
    mcalg = getalgorithm(generator)
    chains = generator.chains
    nchains = length(chains)
    n_tuned_chains = count(c -> c.info.tuned, chains)
    n_converged_chains = count(c -> c.info.converged, chains)

    markdown_append!(md, """
    ### Sample generation

    * Algorithm: MCMC, $(nameof(typeof(mcalg)))
    * MCMC chains: $nchains ($n_tuned_chains tuned, $n_converged_chains converged)
    """)

    return md
end
