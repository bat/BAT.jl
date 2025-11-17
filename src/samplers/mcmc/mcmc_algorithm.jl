# This file is a part of BAT.jl, licensed under the MIT License (MIT).



"""
    abstract type MCMCAlgorithm

Abstract type for Markov chain Monte Carlo algorithms.

To implement a new MCMC algorithm, subtypes of both `MCMCAlgorithm` and
[`MCMCChainState`](@ref) are required.

!!! note

    The details of the `MCMCIterator` and `MCMCAlgorithm` API required to
    implement a new MCMC algorithm currently do not (yet) form part of the
    stable API and are subject to change without deprecation.
"""
abstract type MCMCAlgorithm end
export MCMCAlgorithm



"""
    abstract type MCMCInitAlgorithm

Abstract type for MCMC initialization algorithms.
"""
abstract type MCMCInitAlgorithm end
export MCMCInitAlgorithm

apply_trafo_to_init(f_transform::Function, initalg::MCMCInitAlgorithm) = initalg



"""
    abstract type MCMCProposalTuning

Abstract type for MCMC tuning algorithms.
"""
abstract type MCMCProposalTuning end
export MCMCProposalTuning

"""
    abstract type MCMCProposalTunerState

Abstract type for MCMC tuning algorithm states.
"""
abstract type MCMCProposalTunerState end


"""
    abstract type MCMCTransformTuning

Abstract type for MCMC tuning algorithms.
"""
abstract type MCMCTransformTuning end
export MCMCTransformTuning

"""
    abstract type MCMCTransformTunerState

Abstract type for MCMC tuning algorithm states.
"""
abstract type MCMCTransformTunerState end


"""
    abstract type MCMCTempering

Abstract type for MCMC tempering algorithms.
"""
abstract type MCMCTempering end
export MCMCTempering

"""
    abstract type TemperingState

Abstract type for MCMC tempering algorithm states.
"""
abstract type TemperingState end
export TemperingState


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

BAT.get_samples!(samples::DensitySampleVector, chain::SomeMCMCIter, nonzero_weights::Bool)::typeof(samples)

BAT.next_cycle!(chain::SomeMCMCIter)::SomeMCMCIter

BAT.mcmc_step!!(
    chain::SomeMCMCIter
    callback::Function,
)::nothing
```

The following methods are implemented by default:

```julia
getproposal(chain::MCMCIterator)
mcmc_target(chain::MCMCIterator)
DensitySampleVector(chain::MCMCIterator)
mcmc_iterate!!(chain::MCMCIterator, ...)
mcmc_iterate!!(chains::AbstractVector{<:MCMCIterator}, ...)
isvalidchain(chain::MCMCIterator)
isviablechain(chain::MCMCIterator)
```
"""
abstract type MCMCIterator end
export MCMCIterator

"""
    abstract type MCMCProposal

Abstract type for MCMC proposal algorithms.
"""
abstract type MCMCProposal end

"""
    abstract type MCMCProposalState

Abstract type for MCMC proposal algorithm states.
"""
abstract type MCMCProposalState end

"""
    abstract type SimpleMCMCProposalState

Abstract type for the states of simple MCMC proposal
algorithms, that are implemented in BAT.jl.
This is used to treat more complicated algorithms -that may depend on
external packages- differently.
"""
abstract type SimpleMCMCProposalState <: MCMCProposalState end

"""
    abstract type MCMCBurninAlgorithm

Abstract type for MCMC burn-in algorithms.
"""
abstract type MCMCBurninAlgorithm end
export MCMCBurninAlgorithm


"""
    MCMCState

Carrier type for the states of an MCMC chain, and the states 
of the tuning and tempering algorithms used for sampling.
"""
struct MCMCState{
    C<:MCMCIterator,
    PT<:MCMCProposalTunerState,
    TT<:MCMCTransformTunerState,
    T<:TemperingState
}
    chain_state::C
    proposal_tuner_state::PT
    trafo_tuner_state::TT
    temperer_state::T
end
export MCMCState

"""
    MCMCChainStateInfo

Information about the state of an MCMC chain.
"""
@with_kw struct MCMCChainStateInfo
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

function get_samples! end

function next_cycle! end

function mcmc_step!! end



function mcmc_tuning_init!! end

function mcmc_tuning_postinit!! end

function mcmc_tuning_reinit!! end

function mcmc_tune_transform_post_cycle!! end

function mcmc_tune_post_step!! end

function transform_mcmc_tuning_finalize!! end


function mcmc_init! end

function mcmc_burnin! end


function isvalidstate end

function isviablestate end

function get_target_acceptance_ratio end

get_target_acceptance_ratio(proposal::MCMCProposalState) = proposal.target_acceptance

function get_target_acceptance_int end

get_target_acceptance_int(proposal::MCMCProposalState) = proposal.target_acceptance_int

function mcmc_iterate!! end


function get_proposal_tuning_quality end

# TODO: MD, Think about the exponent in the quality calculation. Should it be user-definable? Where should it be stored?
# Perhaps in the AdaptiveMultiProposalTunerState?
function get_proposal_tuning_quality(
    proposal::MCMCProposalState,
    chain_state::CS,
    beta::Float64
) where CS<:MCMCIterator
    lower, upper = proposal.target_acceptance_int
    eff_acceptance = eff_acceptance_ratio(chain_state)
    target_acceptance = get_target_acceptance_ratio(proposal)

    in_target_interval =  lower < eff_acceptance < upper

    if in_target_interval
        if eff_acceptance >= target_acceptance
            normalization = upper - target_acceptance
            d = (eff_acceptance - target_acceptance) / normalization
        else
            normalization = target_acceptance - lower
            d = (target_acceptance - eff_acceptance) / normalization
        end
        quality = clamp((1 - d)^beta, 0.0, 1.0)
    else
        quality = 0
    end

    return quality
end

function get_tuning_success(
    chain_state::CS,
    proposal::MCMCProposalState
) where CS<:MCMCIterator
    α = eff_acceptance_ratio(chain_state)
    α_min, α_max = get_target_acceptance_int(proposal)
    tuning_success = α_min <= α <= α_max
    return tuning_success
end


# TODO: MD, reincorporate user callback
# TODO: MD, incorporate use of Tempering, so far temperer is not used 
function mcmc_iterate!!(
    output::Union{<:AbstractVector{<:DensitySampleVector},Nothing},
    mcmc_state::MCMCState;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true
)
    @debug "Starting iteration over MCMC chain $(mcmc_state.chain_state.info.id) with $max_nsteps steps in max. $(@sprintf "%.1f s" max_time)"

    start_time = time()
    log_time = start_time
    start_nsteps = nsteps(mcmc_state)
    start_nsamples = nsamples(mcmc_state)
    perform_step = true

    while (perform_step && (time() - start_time) < max_time)
        perform_step = nsteps(mcmc_state) - start_nsteps < max_nsteps
        mcmc_state = perform_step ? mcmc_step!!(mcmc_state) : flush_samples!!(mcmc_state)
        if !isnothing(output)
            get_samples!(output, mcmc_state, nonzero_weights)
        end

        should_log, log_time, elapsed_time = should_log_progress_now(start_time, log_time)
        if should_log
            @debug "Iterating over MCMC chain $(mcmc_state.chain_state.info.id), completed $(nsteps(mcmc_state.chain_state) - start_nsteps) (of $(max_nsteps)) steps and produced $(nsamples(mcmc_state.chain_state) - start_nsamples) samples in $(@sprintf "%.1f s" elapsed_time) so far."
        end
    end

    elapsed_time = time() - start_time
    @debug "Finished iteration over MCMC chain $(mcmc_state.chain_state.info.id), completed $(nsteps(mcmc_state.chain_state) - start_nsteps) steps and produced $(nsamples(mcmc_state.chain_state) - start_nsamples) samples in $(@sprintf "%.1f s" elapsed_time)."

    return mcmc_state
end

function mcmc_iterate!!(
    outputs::Union{AbstractVector{<:AbstractVector{<:DensitySampleVector}}, Nothing},
    mcmc_states::AbstractVector{<:MCMCState};
    kwargs...
)
    if isempty(mcmc_states)
        @debug "No MCMC state(s) to iterate over."
        return mcmc_states
    else
        @debug "Starting iteration over $(length(mcmc_states)) MCMC state(s)"
    end

    outs = isnothing(outputs) ? fill(nothing, size(mcmc_states)...) : outputs
    mcmc_states_new = similar(mcmc_states)

    @sync for i in eachindex(outs, mcmc_states)
        Base.Threads.@spawn mcmc_states_new[i] = mcmc_iterate!!(outs[i], mcmc_states[i]; kwargs...)
    end

    return mcmc_states_new
end

isvalidstate(chain_state::MCMCIterator) = all(current_sample(chain_state).logd .> -Inf)

isviablestate(chain_state::MCMCIterator) = nsamples(chain_state) >= 2

isvalidstate(mcmc_state::MCMCState) = isvalidstate(mcmc_state.chain_state)

isviablestate(mcmc_state::MCMCState) = isvalidstate(mcmc_state.chain_state)


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
    chain_states::T
end

function MCMCSampleGenerator(mcmc_states::AbstractVector{<:MCMCState})
    MCMCSampleGenerator(getfield.(mcmc_states, :chain_state))
end


getproposal(sg::MCMCSampleGenerator) = sg.chain_states[1].proposal


function Base.show(io::IO, generator::MCMCSampleGenerator)
    if get(io, :compact, false)
        print(io, nameof(typeof(generator)), "(")
        if !isempty(generator.chain_states)
            show(io, first(generator.chain_states))
            print(io, ", …")
        end
        print(io, ")")
    else
        println(io, nameof(typeof(generator)), ":")
        chain_states = generator.chain_states
        n_chain_states = length(chain_states)
        n_tuned_chain_states = count(c -> c.info.tuned, chain_states)
        n_converged_chain_states = count(c -> c.info.converged, chain_states)
        print(io, "proposal: ")
        show(io, "text/plain", getproposal(generator))
        println(io, "number of chains:", repeat(' ', 13), n_chain_states)
        println(io, "number of chains tuned:", repeat(' ', 7), n_tuned_chain_states)
        println(io, "number of chains converged:", repeat(' ', 3), n_converged_chain_states)
        print(io, "number of samples per chain:", repeat(' ', 2), nsamples(chain_states[1]))
    end
end


function LazyReports.pushcontent!(rpt::LazyReport, generator::MCMCSampleGenerator)
    mcalg = getproposal(generator)
    chain_states = generator.chain_states
    n_chain_states = length(chain_states)
    n_tuned_chain_states = count(c -> c.info.tuned, chain_states)
    n_converged_chain_states = count(c -> c.info.converged, chain_states)

    lazyreport!(rpt, """
    ### Sample generation

    * Algorithm: MCMC, $(nameof(typeof(mcalg)))
    * MCMC chains: $n_chain_states ($n_tuned_chain_states tuned, $n_converged_chain_states converged)
    """)

    return nothing
end
