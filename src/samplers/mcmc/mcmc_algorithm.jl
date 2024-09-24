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
    abstract type MCMCTuning

Abstract type for MCMC tuning algorithms.
"""
abstract type MCMCTuning end
export MCMCTuning


"""
    abstract type MCMCTempering
Abstract type for MCMC tempering algorithms.
"""
abstract type MCMCTempering end
export MCMCTempering

abstract type AbstractMCMCTemperingInstance end
export AbstractMCMCTemperingInstance

"""
    abstract type MCMCBurninAlgorithm

Abstract type for MCMC burn-in algorithms.
"""
abstract type MCMCBurninAlgorithm end
export MCMCBurninAlgorithm



# TODO: MD, adjust doctring for new typestructure
"""
    abstract type MCMCIterator end

*BAT-internal, not part of stable public API.*

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
BAT.getproposal(chain::SomeMCMCIter)::MCMCAlgorithm

BAT.mcmc_target(chain::SomeMCMCIter)::BATMeasure

BAT.get_context(chain::SomeMCMCIter)::BATContext

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
getproposal(chain::MCMCIterator)
mcmc_target(chain::MCMCIterator)
DensitySampleVector(chain::MCMCIterator)
mcmc_iterate!(chain::MCMCIterator, ...)
mcmc_iterate!(chains::AbstractVector{<:MCMCIterator}, ...)
isvalidchain(chain::MCMCIterator)
isviablechain(chain::MCMCIterator)
```
"""
abstract type MCMCIterator end
export MCMCIterator


# TODO: MD, adjust doctring for new typestructure
abstract type MCMCProposal end

abstract type MCMCProposalState end

@with_kw struct MCMCStateInfo
    id::Int32
    cycle::Int32
    tuned::Bool
    converged::Bool
end


function Base.show(io::IO, mc_state::MCMCIterator)
    print(io, Base.typename(typeof(mc_state)).name, "(")
    print(io, "id = "); show(io, mcmc_info(mc_state).id)
    print(io, ", nsamples = "); show(io, nsamples(mc_state))
    print(io, ", target = "); show(io, mcmc_target(mc_state))
    print(io, ")") 
end


function getproposal end

function mcmc_target end

function mcmc_info end

function nsteps end

function nsamples end

function current_sample end

function sample_type end

function samples_available end

function get_samples! end

function next_cycle! end

function mcmc_step! end


abstract type AbstractMCMCTunerInstance end


function tuning_init! end

function tuning_postinit! end

function tuning_reinit! end

function tuning_update! end

function tune_transform!! end

function tuning_finalize! end

function tuning_callback end


function mcmc_init! end

function mcmc_burnin! end


function isvalidstate end

function isviablestate end


function mcmc_iterate! end

# TODO: MD, incorporate use of Tempering, so far temperer is not used 
function mcmc_iterate!(
    output::Union{DensitySampleVector,Nothing},
    mc_state::MCMCIterator,
    tuner::Union{AbstractMCMCTunerInstance,Nothing},
    temperer::Union{AbstractMCMCTemperingInstance,Nothing};
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func
)
    @debug "Starting iteration over MCMC chain $(mc_state.info.id) with $max_nsteps steps in max. $(@sprintf "%.1f s" max_time)"

    start_time = time()
    last_progress_message_time = start_time
    start_nsteps = nsteps(mc_state)
    start_nsamples = nsamples(mc_state)

    while (
        (nsteps(mc_state) - start_nsteps) < max_nsteps &&
        (time() - start_time) < max_time
    )
        mcmc_step!(mc_state, tuner, temperer)
        callback(Val(:mcmc_step), mc_state)
        if !isnothing(output)
            get_samples!(output, mc_state, nonzero_weights)
        end
        current_time = time()
        elapsed_time = current_time - start_time
        logging_interval = 5 * round(log2(elapsed_time/60 + 1) + 1)
        if current_time - last_progress_message_time > logging_interval
            last_progress_message_time = current_time
            @debug "Iterating over MCMC chain $(mc_state.info.id), completed $(nsteps(mc_state) - start_nsteps) (of $(max_nsteps)) steps and produced $(nsamples(mc_state) - start_nsamples) samples in $(@sprintf "%.1f s" elapsed_time) so far."
        end
    end

    current_time = time()
    elapsed_time = current_time - start_time
    @debug "Finished iteration over MCMC chain $(mc_state.info.id), completed $(nsteps(mc_state) - start_nsteps) steps and produced $(nsamples(mc_state) - start_nsamples) samples in $(@sprintf "%.1f s" elapsed_time)."

    return nothing
end


function mcmc_iterate!(
    output::Union{DensitySampleVector,Nothing},
    mc_state::MCMCIterator;
    tuner::Union{AbstractMCMCTunerInstance, Nothing} = nothing,
    temperer::Union{AbstractMCMCTemperingInstance, Nothing} = nothing,
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    callback::Function = nop_func
)
    #cb = combine_callbacks(tuning_callback(tuner), callback)
    mcmc_iterate!(
        output, mc_state, tuner, temperer;
        max_nsteps = max_nsteps, max_time = max_time, nonzero_weights = nonzero_weights, callback = callback
    )

    return nothing
end


function mcmc_iterate!(
    outputs::Union{AbstractVector{<:DensitySampleVector},Nothing},
    mc_states::AbstractVector{<:MCMCIterator};
    tuners::Union{AbstractVector{<:AbstractMCMCTunerInstance},Nothing} = nothing,
    temperers::Union{AbstractVector{<:AbstractMCMCTemperingInstance},Nothing} = nothing,
    kwargs...
)
    if isempty(mc_states)
        @debug "No MCMC chain(s) to iterate over."
        return mc_states
    else
        @debug "Starting iteration over $(length(mc_states)) MCMC chain(s)"
    end

    outs = isnothing(outputs) ? fill(nothing, size(mc_states)...) : outputs
    tnrs = isnothing(tuners) ? fill(nothing, size(mc_states)...) : tuners
    tmrs = isnothing(temperers) ? fill(nothing, size(mc_states)...) : temperers

    @sync for i in eachindex(outs, mc_states, tnrs)
        Base.Threads.@spawn mcmc_iterate!(outs[i], mc_states[i]; tuner = tnrs[i], temperer = tmrs[i], kwargs...)
    end

    return nothing
end


isvalidstate(state::MCMCIterator) = current_sample(state).logd > -Inf

isviablestate(mc_state::MCMCIterator) = nsamples(mc_state) >= 2



"""
    BAT.MCMCSampleGenerator

*BAT-internal, not part of stable public API.*

MCMC sample generator.

Constructors:

```julia
MCMCSampleGenerator(mc_state::AbstractVector{<:MCMCIterator})
```
"""
struct MCMCSampleGenerator{T<:AbstractVector{<:MCMCIterator}} <: AbstractSampleGenerator
    mc_states::T
end

getproposal(sg::MCMCSampleGenerator) = sg.mc_states[1].proposal


function Base.show(io::IO, generator::MCMCSampleGenerator)
    if get(io, :compact, false)
        print(io, nameof(typeof(generator)), "(")
        if !isempty(generator.mc_states)
            show(io, first(generator.mc_states))
            print(io, ", …")
        end
        print(io, ")")
    else
        println(io, nameof(typeof(generator)), ":")
        mc_states = generator.mc_states
        n_mc_states = length(mc_states)
        n_tuned_mc_states = count(c -> c.info.tuned, mc_states)
        n_converged_mc_states = count(c -> c.info.converged, mc_states)
        print(io, "proposal: ")
        show(io, "text/plain", getproposal(generator))
        println(io, "number of chains:", repeat(' ', 13), n_mc_states)
        println(io, "number of chains tuned:", repeat(' ', 7), n_tuned_mc_states)
        println(io, "number of chains converged:", repeat(' ', 3), n_converged_mc_states)
        print(io, "number of samples per chain:", repeat(' ', 2), nsamples(mc_states[1]))
    end
end


function bat_report!(md::Markdown.MD, generator::MCMCSampleGenerator)
    mcalg = getproposal(generator)
    mc_states = generator.mc_states
    n_mc_states = length(mc_states)
    n_tuned_mc_states = count(c -> c.info.tuned, mc_states)
    n_converged_mc_states = count(c -> c.info.converged, mc_states)

    markdown_append!(md, """
    ### Sample generation

    * Algorithm: MCMC, $(nameof(typeof(mcalg)))
    * MCMC chains: $n_mc_states ($n_tuned_mc_states tuned, $n_converged_mc_states converged)
    """)

    return md
end
