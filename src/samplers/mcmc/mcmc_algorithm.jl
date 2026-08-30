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

*Experimental feature, not part of stable public API.*

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

*Experimental feature, not part of stable public API.*

Abstract type for MCMC tuning algorithm states.
"""
abstract type MCMCTransformTunerState end


"""
    abstract type MCMCTempering

*Experimental feature, not part of stable public API.*

Abstract type for MCMC tempering algorithms.
"""
abstract type MCMCTempering end
export MCMCTempering

"""
    abstract type TemperingState

*Experimental feature, not part of stable public API.*

Abstract type for MCMC tempering algorithm states.
"""
abstract type TemperingState end
export TemperingState


# TODO: MD, adjust doctring for new typestructure
"""
    abstract type MCMCIterator end

*Experimental feature, not part of stable public API.*

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

*Experimental feature, not part of stable public API.*

Abstract type for MCMC proposal algorithms.
"""
abstract type MCMCProposal end

"""
    abstract type MCMCProposalState

*Experimental feature, not part of stable public API.*

Abstract type for MCMC proposal algorithm states.
"""
abstract type MCMCProposalState end

_validate_mcmc_proposal_configuration(
    ::MCMCProposal,
    ::MCMCProposalTuning,
) = nothing

_validate_mcmc_weighting_configuration(
    ::MCMCProposal,
    ::AbstractMCMCWeightingScheme,
) = nothing

_validate_mcmc_adaptive_transform_configuration(
    ::MCMCProposal,
    ::AbstractAdaptiveTransform,
) = nothing

_validate_mcmc_proposal_tuning_configuration(
    ::MCMCProposal,
    ::MCMCProposalTuning,
) = nothing

_unsupported_mcmc_component_tuning(proposal, tuning) = throw(ArgumentError(
    "Unsupported MCMC component tuning pair: $(nameof(typeof(proposal))) with $(nameof(typeof(tuning)))",
))

function _acceptance_in_target(proposal, acceptance)
    lower, upper = get_target_acceptance_int(proposal)
    return lower <= acceptance <= upper
end

"""
    abstract type SimpleMCMCProposalState

*Experimental feature, not part of stable public API.*

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

*Experimental feature, not part of stable public API.*

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

*Experimental feature, not part of stable public API.*

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

function mcmc_tune_post_step!! end


"""
    struct BAT.MCMCStepInfo

*BAT-internal, not part of stable public API.*

Per-walker information about one MCMC transition, produced by
`mcmc_propose!!` and consumed by the tuning machinery and sample
weighting. `p_accept` is always present; gradient-based proposals
additionally provide the z-space log-density gradients at the selected
states (`z_grads`) and trajectory diagnostics (`divergent`, `tree_depth`,
`n_leapfrog`), which are `nothing` for proposals that don't compute them.
`walker_order` indexes these vectors in logical-walker order.
"""
struct MCMCStepInfo{
    PA<:AbstractVector{<:Real},
    GS<:Union{Nothing,AbstractVector{<:AbstractVector{<:Real}}},
    DV<:Union{Nothing,AbstractVector{Bool}},
    IV<:Union{Nothing,AbstractVector{<:Integer}},
    WO<:AbstractVector{<:Integer}
}
    p_accept::PA
    z_grads::GS
    divergent::DV
    tree_depth::IV
    n_leapfrog::IV
    walker_order::WO
end

MCMCStepInfo(p_accept::AbstractVector{<:Real}, z_grads, divergent, tree_depth, n_leapfrog) =
    MCMCStepInfo(p_accept, z_grads, divergent, tree_depth, n_leapfrog, eachindex(p_accept))

MCMCStepInfo(p_accept::AbstractVector{<:Real}) =
    MCMCStepInfo(p_accept, nothing, nothing, nothing, nothing)


# Whether a proposal state provides z-space log-density gradients in its
# MCMCStepInfo (required by gradient-based transform tuners):
mcmc_step_provides_grads(::MCMCProposalState) = false


# Transform-tuner state creation may take the declared adaptive transform
# into account (e.g. to match the estimation structure to the transform
# structure); by default it is ignored:
function create_trafo_tuner_state end

function create_trafo_tuner_state(
    tuning::MCMCTransformTuning,
    chain_state::CS,
    n_steps_hint::Integer,
    ::AbstractAdaptiveTransform
) where CS<:MCMCIterator
    return create_trafo_tuner_state(tuning, chain_state, n_steps_hint)
end


# Whether a transform change installed by this transform tuner should
# restart step-size adaptation (with a fresh reasonable-step-size search).
# True for tuners that commit discrete geometry changes (windowed or
# drift-committed schedules); tuners that drift the transform continuously
# in small per-step updates (like RAM) return false, step-size adaptation
# simply tracks them:
transform_change_restarts_stepsize(::MCMCTransformTunerState) = true

transform_tuning_pauses_proposal(::MCMCTransformTunerState) = false

# Called by the tuning orchestration instead of mcmc_tune_proposal_post_step!!
# when a transform tuner has installed a new transformation and its policy
# requests step-size readaptation. The step statistic that crossed the
# geometry change is discarded (not passed on):
function mcmc_proposal_transform_committed!!(
    proposal::MCMCProposalState,
    tuner::MCMCProposalTunerState,
    chain_state::CS
) where CS<:MCMCIterator
    return proposal, tuner, chain_state
end

mcmc_proposal_transform_committed!!(
    proposal::MCMCProposalState,
    tuner::MCMCProposalTunerState,
    chain_state::CS,
    ::MCMCTransformTunerState,
) where {CS<:MCMCIterator} =
    mcmc_proposal_transform_committed!!(proposal, tuner, chain_state)


function mcmc_trafo_tuning_init!! end

function mcmc_trafo_tuning_postinit!! end

function mcmc_trafo_tuning_reinit!! end

function mcmc_tune_trafo_post_step!! end


function mcmc_proposal_tuning_init!! end

function mcmc_proposal_tuning_postinit!! end

function mcmc_proposal_tuning_reinit!! end

function mcmc_tune_proposal_post_step!! end


function mcmc_init! end

function mcmc_burnin! end


function isvalidstate end

function isviablestate end


function mcmc_trafo_tuning_init!!(
    ::MCMCTransformTunerState,
    ::CS,
    ::Integer
) where CS<:MCMCIterator
    return nothing
end

function mcmc_trafo_tuning_reinit!!(
    ::MCMCTransformTunerState,
    ::CS,
    ::Integer
) where CS<:MCMCIterator
    return nothing
end

function mcmc_trafo_tuning_postinit!!(
    tuner::MCMCTransformTunerState,
    chain_state::CS,
    samples::AbstractVector{<:DensitySampleVector}
) where CS<:MCMCIterator
    return nothing
end

function mcmc_tune_trafo_post_cycle!!(
    f_transform::Function,
    tuner::MCMCTransformTunerState,
    chain_state::CS,
    proposal::MCMCProposalState,
    samples::AbstractVector{<:DensitySampleVector}
) where CS<:MCMCIterator
    return f_transform, tuner, chain_state
end

function mcmc_trafo_tuning_finalize!!(
    f_transform::Function,
    trafo_tuner_state::MCMCTransformTunerState,
    chain_state::CS
) where CS<:MCMCIterator
    return f_transform, trafo_tuner_state, chain_state
end

function mcmc_tune_trafo_post_step!!(
    f_transform::Function,
    tuner::MCMCTransformTunerState,
    chain_state::CS,
    ::MCMCProposalState,
    ::NamedTuple,
    ::NamedTuple,
    ::MCMCStepInfo
) where CS<:MCMCIterator
    return f_transform, tuner, chain_state
end


function mcmc_proposal_tuning_init!!(
    ::MCMCProposalTunerState,
    ::CS,
    ::Integer
) where CS<:MCMCIterator
    return nothing
end

function mcmc_proposal_tuning_reinit!!(
    ::MCMCProposalTunerState,
    ::CS,
    ::Integer
) where CS<:MCMCIterator
    return nothing
end

function mcmc_proposal_tuning_postinit!!(
    ::MCMCProposalTunerState,
    ::CS,
    ::AbstractVector{<:DensitySampleVector}
) where CS<:MCMCIterator
    return nothing
end

function mcmc_tune_proposal_post_cycle!!(
    proposal::MCMCProposalState, 
    tuner::MCMCProposalTunerState, 
    chain_state::CS, 
    ::AbstractVector{<:DensitySampleVector}
) where CS<:MCMCIterator
    return proposal, tuner, chain_state
end

function mcmc_proposal_tuning_finalize!!(
    proposal_state::MCMCProposalState,
    proposal_tuner_state::MCMCProposalTunerState,
    chain_state::CS
) where CS<:MCMCIterator
    return proposal_state, proposal_tuner_state, chain_state
end

# Marks the warmup/retained-sampling boundary in per-chain diagnostics,
# called after all of warmup (tuning and post-tuning stabilization), not
# at tuning finalization. Mutates shared diagnostics objects in place, so
# functional proposal-state updates are unaffected:
mcmc_mark_warmup_end!(::MCMCProposalState) = nothing

function mcmc_tune_proposal_post_step!!(
    proposal::MCMCProposalState,
    tuner::MCMCProposalTunerState,
    chain_state::CS,
    ::MCMCStepInfo
) where CS<:MCMCIterator
    return proposal, tuner, chain_state
end


function get_target_acceptance_ratio(proposal::MCMCProposalState)
   return proposal.target_acceptance
end

function get_target_acceptance_int(proposal::MCMCProposalState) 
    return proposal.target_acceptance_int
end

function mcmc_iterate!! end


function get_proposal_tuning_quality end

# TODO: MD, Think about the exponent in the quality calculation. Should it be user-definable? Where should it be stored?
# Perhaps in the AdaptiveMultiProposalTunerState?
get_proposal_tuning_quality(
    proposal::MCMCProposalState,
    chain_state::CS,
    beta::Float64
) where CS<:MCMCIterator = get_proposal_tuning_quality(
    proposal, eff_acceptance_ratio(chain_state), beta,
)

function get_proposal_tuning_quality(
    proposal::MCMCProposalState,
    eff_acceptance::Real,
    beta::Float64,
)
    lower, upper = proposal.target_acceptance_int
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
        quality = 0.0
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

# Proposal tuners that track statistics of the current tuning cycle can
# specialize the three-argument form and judge tuning success on those
# (see e.g. the HMC step-size adaptor); by default the tuner state is
# ignored. Note that eff_acceptance_ratio is a state-movement rate, which
# coincides with the mean acceptance probability only for accept/reject
# proposals like random walk Metropolis:
function get_tuning_success(
    chain_state::CS,
    proposal::MCMCProposalState,
    ::MCMCProposalTunerState
) where CS<:MCMCIterator
    return get_tuning_success(chain_state, proposal)
end

get_active_proposal_idx(proposal_state::MCMCProposalState) = 1

function next_proposal!!(
    rng::AbstractRNG,
    proposal_state::MCMCProposalState,
    stepno::Integer
)
    return proposal_state, proposal_state
end

function get_active_proposal(
    proposal_state::MCMCProposalState,
)
    return proposal_state
end

function update_active_proposal!!(
    proposal::MCMCProposalState,
    active_proposal_new::MCMCProposalState
)
    return proposal    
end

# TODO: MD, reincorporate user callback
# TODO: MD, incorporate use of Tempering, so far temperer is not used 
function mcmc_iterate!!(
    output::Union{<:AbstractVector{<:DensitySampleVector},Nothing},
    mcmc_state::MCMCState;
    max_nsteps::Integer = 1,
    max_time::Real = Inf,
    nonzero_weights::Bool = true,
    _cancelled::Union{Nothing,Base.Threads.Atomic{Bool}} = nothing,
)
    @debug "Starting iteration over MCMC chain $(mcmc_state.chain_state.info.id) with $max_nsteps steps in max. $(@sprintf "%.1f s" max_time)"

    start_time = time()
    log_time = start_time
    start_nsteps = nsteps(mcmc_state)
    start_nsamples = nsamples(mcmc_state)
    perform_step = true

    while (perform_step && (time() - start_time) < max_time)
        if !isnothing(_cancelled) && _cancelled[]
            mcmc_state = flush_samples!!(mcmc_state)
            isnothing(output) || get_samples!(output, mcmc_state, nonzero_weights)
            break
        end
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
    # Tuning may change type parameters of the states (e.g. the structural
    # type of an adaptive transform on its first commit), so the result
    # container must not be bound to the input element type:
    mcmc_states_new = similar(mcmc_states, MCMCState)
    cancelled = Base.Threads.Atomic{Bool}(false)

    @sync for i in eachindex(outs, mcmc_states)
        Base.Threads.@spawn try
            mcmc_states_new[i] = mcmc_iterate!!(outs[i], mcmc_states[i];
                kwargs..., _cancelled = cancelled)
        catch
            cancelled[] = true
            rethrow()
        end
    end

    return mcmc_states_new
end

isvalidstate(chain_state::MCMCIterator) = all(current_sample(chain_state).logd .> -Inf)

isviablestate(chain_state::MCMCIterator) = nsamples(chain_state) >= 2

isvalidstate(mcmc_state::MCMCState) = isvalidstate(mcmc_state.chain_state)

isviablestate(mcmc_state::MCMCState) = isviablestate(mcmc_state.chain_state)


"""
    BAT.MCMCSampleGenerator

*BAT-internal, not part of stable public API.*

MCMC sample generator, holds the (mutable) states of the MCMC chains.
Consumers must not mutate the chain states, continuing sample generation
requires a deep copy.

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
