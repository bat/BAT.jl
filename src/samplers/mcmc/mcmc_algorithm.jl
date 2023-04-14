# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type MCMCAlgorithm

Abstract type for Markov chain Monte Carlo algorithms.

To implement a new MCMC algorithm, subtypes of both `MCMCAlgorithm` and
[`MCMCIterator`](@ref) are required.

!!! note

    The details of the `MCMCIterator` and `MCMCAlgorithm` API required to
    implement a new MCMC algorithm currently do not (yet) form part of the
    stable API and are subject to change without deprecation.
"""
abstract type MCMCAlgorithm end
export MCMCAlgorithm


function get_mcmc_tuning end


"""
    abstract type MCMCInitAlgorithm

Abstract type for MCMC initialization algorithms.
"""
abstract type MCMCInitAlgorithm end
export MCMCInitAlgorithm

apply_trafo_to_init(trafo::Function, initalg::MCMCInitAlgorithm) = initalg



"""
    abstract type MCMCTuningAlgorithm

Abstract type for MCMC tuning algorithms.
"""
abstract type MCMCTuningAlgorithm end
export MCMCTuningAlgorithm



"""
    abstract type MCMCBurninAlgorithm

Abstract type for MCMC burn-in algorithms.
"""
abstract type MCMCBurninAlgorithm end
export MCMCBurninAlgorithm



@with_kw struct MCMCIteratorInfo
    id::Int32
    cycle::Int32
    tuned::Bool
    converged::Bool
end


"""
    abstract type MCMCIterator end

Represents the current state of an MCMC chain.

!!! note

    The details of the `MCMCIterator` and `MCMCAlgorithm` API (see below)
    currently do not form part of the stable API and are subject to change
    without deprecation.

To implement a new MCMC algorithm, subtypes of both [`MCMCAlgorithm`](@ref)
and `MCMCIterator` are required.

The following methods must be defined for subtypes of `MCMCIterator` (e.g.
`SomeMCMCIter<:MCMCIterator`):

```julia
BAT.getalgorithm(chain::SomeMCMCIter)::MCMCAlgorithm

BAT.getmeasure(chain::SomeMCMCIter)::AbstractMeasureOrDensity

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
getmeasure(chain::MCMCIterator)
DensitySampleVector(chain::MCMCIterator)
mcmc_iterate!(chain::MCMCIterator, ...)
mcmc_iterate!(chains::AbstractVector{<:MCMCIterator}, ...)
isvalidchain(chain::MCMCIterator)
isviablechain(chain::MCMCIterator)
```
"""
abstract type MCMCIterator end
export MCMCIterator


function Base.show(io::IO, chain::MCMCIterator)
    print(io, Base.typename(typeof(chain)).name, "(")
    print(io, "id = "); show(io, mcmc_info(chain).id)
    print(io, ", nsamples = "); show(io, nsamples(chain))
    print(io, ", density = "); show(io, getmeasure(chain))
    print(io, ")") 
end


function getalgorithm end

function getmeasure end

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



DensitySampleVector(chain::MCMCIterator) = DensitySampleVector(sample_type(chain), totalndof(getmeasure(chain)))



abstract type AbstractMCMCTunerInstance end


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


function mcmc_iterate!(
    output::Union{DensitySampleVector,Nothing},
    chain::MCMCIterator,
    tuner::Nothing = nothing;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func
)
    @debug "Starting iteration over MCMC chain $(chain.info.id) with $max_nsteps steps in max. $(@sprintf "%.1f s" max_time)"

    start_time = time()
    last_progress_message_time = start_time
    start_nsteps = nsteps(chain)
    start_nsamples = nsamples(chain)

    while (
        (nsteps(chain) - start_nsteps) < max_nsteps &&
        (time() - start_time) < max_time
    )
        mcmc_step!(chain)
        callback(Val(:mcmc_step), chain)
        if !isnothing(output)
            get_samples!(output, chain, nonzero_weights)
        end
        current_time = time()
        elapsed_time = current_time - start_time
        logging_interval = 5 * round(log2(elapsed_time/60 + 1) + 1)
        if current_time - last_progress_message_time > logging_interval
            last_progress_message_time = current_time
            @debug "Iterating over MCMC chain $(chain.info.id), completed $(nsteps(chain) - start_nsteps) (of $(max_nsteps)) steps and produced $(nsamples(chain) - start_nsamples) samples in $(@sprintf "%.1f s" elapsed_time) so far."
        end
    end

    current_time = time()
    elapsed_time = current_time - start_time
    @debug "Finished iteration over MCMC chain $(chain.info.id), completed $(nsteps(chain) - start_nsteps) steps and produced $(nsamples(chain) - start_nsamples) samples in $(@sprintf "%.1f s" elapsed_time)."

    return nothing
end


function mcmc_iterate!(
    output::Union{DensitySampleVector,Nothing},
    chain::MCMCIterator,
    tuner::AbstractMCMCTunerInstance;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func
)
    cb = combine_callbacks(tuning_callback(tuner), callback)
    mcmc_iterate!(
        output, chain;
        max_nsteps = max_nsteps, max_time = max_time, nonzero_weights = nonzero_weights, callback = cb
    )

    return nothing
end


function mcmc_iterate!(
    outputs::Union{AbstractVector{<:DensitySampleVector},Nothing},
    chains::AbstractVector{<:MCMCIterator},
    tuners::Union{AbstractVector{<:AbstractMCMCTunerInstance},Nothing} = nothing;
    kwargs...
)
    if isempty(chains)
        @debug "No MCMC chain(s) to iterate over."
        return chains
    else
        @debug "Starting iteration over $(length(chains)) MCMC chain(s)"
    end

    outs = isnothing(outputs) ? fill(nothing, size(chains)...) : outputs
    tnrs = isnothing(tuners) ? fill(nothing, size(chains)...) : tuners

    @sync for i in eachindex(outs, chains, tnrs)
        Base.Threads.@spawn mcmc_iterate!(outs[i], chains[i], tnrs[i]; kwargs...)
    end

    return nothing
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

getalgorithm(sg::MCMCSampleGenerator) = sg.chains[1].algorithm


function Base.show(io::IO, generator::MCMCSampleGenerator)
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
        println(io, "number of chains:", repeat(' ', 13), nchains)
        println(io, "number of chains tuned:", repeat(' ', 7), n_tuned_chains)
        println(io, "number of chains converged:", repeat(' ', 3), n_converged_chains)
        print(io, "number of samples per chain:", repeat(' ', 2), nsamples(chains[1]))
    end
end




function bat_report!(md::Markdown.MD, generator::MCMCSampleGenerator)
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
