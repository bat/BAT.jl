# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct MultiProposalTuning <: MCMCProposalTuning

Tuning algorithm for MCMCMultiProposals.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
struct MultiProposalTuning{
    PT<:Vector{<:MCMCProposalTuning},
}<:MCMCProposalTuning
    proposal_tunings::PT
end

export MultiProposalTuning

function _validate_mcmc_proposal_tuning_configuration(
    multi_proposal::MCMCMultiProposal,
    tuning::NoMCMCProposalTuning,
)
    for proposal in multi_proposal.proposals
        proposal isa HamiltonianMC &&
            _unsupported_mcmc_component_tuning(proposal, tuning)
    end
    return nothing
end

_validate_mcmc_proposal_tuning_configuration(
    multi_proposal::MCMCMultiProposal,
    tuning::HMCTuning,
) = _unsupported_mcmc_component_tuning(multi_proposal, tuning)

function _validate_mcmc_proposal_tuning_configuration(
    multi_proposal::MCMCMultiProposal,
    tuning::MultiProposalTuning,
)
    n_proposals = length(multi_proposal.proposals)
    n_tunings = length(tuning.proposal_tunings)
    n_tunings == n_proposals || throw(ArgumentError(
        "MultiProposalTuning has $n_tunings component tunings but MCMCMultiProposal has $n_proposals component proposals",
    ))
    for (proposal, component_tuning) in zip(
        multi_proposal.proposals, tuning.proposal_tunings,
    )
        if proposal isa HamiltonianMC && component_tuning isa NoMCMCProposalTuning
            _unsupported_mcmc_component_tuning(proposal, component_tuning)
        end
        _validate_mcmc_proposal_configuration(proposal, component_tuning)
    end
    return nothing
end

struct MultiProposalTunerState{
    PTS<:Vector{<:MCMCProposalTunerState},
}<:MCMCProposalTunerState
    proposal_tuners::PTS
end




function create_proposal_tuner_state(
    multi_tuning::MultiProposalTuning, 
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
    iteration::Integer
)
    proposal_tuners_init = Vector{MCMCProposalTunerState}()

    proposal_tunings = multi_tuning.proposal_tunings
    proposals = multi_proposal.proposal_states

    for i in eachindex(multi_tuning.proposal_tunings)
        tuner_tmp = create_proposal_tuner_state(
            proposal_tunings[i],
            chain_state,
            proposals[i],
            iteration
        )

        push!(proposal_tuners_init, tuner_tmp)
    end

    return MultiProposalTunerState(proposal_tuners_init)
end

function mcmc_proposal_tuning_init!!(
    multi_tuner_state::MultiProposalTunerState, 
    chain_state::MCMCChainState, 
    max_nsteps::Integer
)
    for tuner in multi_tuner_state.proposal_tuners
        mcmc_proposal_tuning_init!!(tuner, chain_state, max_nsteps)
    end
end

function mcmc_proposal_tuning_reinit!!(
    multi_tuner_state::MultiProposalTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    for tuner in multi_tuner_state.proposal_tuners
        mcmc_proposal_tuning_reinit!!(tuner, chain_state, max_nsteps)
    end
end


function mcmc_proposal_tuning_postinit!!(
    multi_tuner::MultiProposalTunerState, 
    chain_state::MCMCChainState, 
    samples::AbstractVector{<:DensitySampleVector}
)
    for tuner in multi_tuner.proposal_tuners
        mcmc_proposal_tuning_postinit!!(tuner, chain_state, samples)
    end
end


function mcmc_tune_proposal_post_cycle!!(
    multi_proposal::MultiProposalState,
    multi_tuner::MultiProposalTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    proposals = multi_proposal.proposal_states
    tuners = multi_tuner.proposal_tuners
    for i in eachindex(proposals)
        proposals[i], tuners[i], chain_state = mcmc_tune_proposal_post_cycle!!(
            proposals[i],
            tuners[i],
            chain_state,
            samples
        )
    end

    return multi_proposal, multi_tuner, chain_state 
end


function mcmc_proposal_tuning_finalize!!(
    multi_proposal::MultiProposalState,
    multi_tuner::MultiProposalTunerState, 
    chain_state::MCMCChainState
)
    proposals = multi_proposal.proposal_states
    tuners = multi_tuner.proposal_tuners
    for i in eachindex(proposals)
        proposals[i], tuners[i], chain_state = mcmc_proposal_tuning_finalize!!(
            proposals[i], tuners[i], chain_state,
        )
    end

    return multi_proposal, multi_tuner, chain_state
end

function mcmc_tune_proposal_post_step!!(
    multi_proposal::MultiProposalState,
    multi_tuner::MultiProposalTunerState,
    chain_state::MCMCChainState,
    step_info::MCMCStepInfo
)
    active_idx = multi_proposal.active_idx
    
    active_proposal = get_active_proposal(multi_proposal)
    active_tuner = multi_tuner.proposal_tuners[active_idx]

    active_proposal_tuned, active_tuner, chain_state = mcmc_tune_proposal_post_step!!(
        active_proposal, 
        active_tuner, 
        chain_state, 
        step_info
    )

    multi_proposal = update_active_proposal!!(multi_proposal, active_proposal_tuned)
    multi_tuner.proposal_tuners[active_idx] = active_tuner

    return multi_proposal, multi_tuner, chain_state
end

function get_tuning_success(
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
    multi_tuner::MultiProposalTunerState,
)
    proposals = multi_proposal.proposal_states
    tuners = multi_tuner.proposal_tuners
    acceptance_rates = detailed_eff_acceptance_ratio(chain_state)
    return all(eachindex(proposals)) do i
        _component_tuning_success(chain_state, proposals[i], tuners[i], acceptance_rates[i])
    end
end

_component_tuning_success(chain_state, proposal, tuner, acceptance) =
    get_tuning_success(chain_state, proposal, tuner)

_component_tuning_success(
    chain_state, proposal, ::NoMCMCProposalTunerState, acceptance,
) = _acceptance_in_target(proposal, acceptance)

_component_tuning_success(
    chain_state, proposal::MALAProposalState, tuner::MALAStepSizeTunerState, acceptance,
) = tuner.min_run_nobs == 0 ?
    _acceptance_in_target(proposal, acceptance) :
    get_tuning_success(chain_state, proposal, tuner)
