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
    PT<:Tuple{Vararg{MCMCProposalTuning}},
}<:MCMCProposalTuning
    proposal_tunings::PT
end

MultiProposalTuning(tunings::Vector{<:MCMCProposalTuning}) = MultiProposalTuning(Tuple(tunings))

export MultiProposalTuning

function _validate_mcmc_proposal_tuning_configuration(
    multi_proposal::MCMCMultiProposal,
    tuning::NoMCMCProposalTuning,
)
    # Stop at the first invalid proposal.
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
    # Stop at the first invalid pair.
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
    PTS<:Tuple{Vararg{MCMCProposalTunerState}},
}<:MCMCProposalTunerState
    proposal_tuners::PTS
end




function create_proposal_tuner_state(
    multi_tuning::MultiProposalTuning, 
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
    iteration::Integer
)
    tuners = create_proposal_tuner_state.(
        multi_tuning.proposal_tunings, Ref(chain_state), multi_proposal.proposal_states, iteration,
    )
    return MultiProposalTunerState(tuners)
end

function mcmc_proposal_tuning_init!!(
    multi_tuner_state::MultiProposalTunerState, 
    chain_state::MCMCChainState, 
    max_nsteps::Integer
)
    for i in eachindex(multi_tuner_state.proposal_tuners)
        component_chain = @set chain_state.proposal = chain_state.proposal.proposal_states[i]
        mcmc_proposal_tuning_init!!(multi_tuner_state.proposal_tuners[i], component_chain, max_nsteps)
    end
end

function mcmc_proposal_tuning_reinit!!(
    multi_tuner_state::MultiProposalTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    for i in eachindex(multi_tuner_state.proposal_tuners)
        component_chain = @set chain_state.proposal = chain_state.proposal.proposal_states[i]
        mcmc_proposal_tuning_reinit!!(multi_tuner_state.proposal_tuners[i], component_chain, max_nsteps)
    end
end


function mcmc_proposal_transform_committed!!(
    multi_proposal::MultiProposalState,
    multi_tuner::MultiProposalTunerState,
    chain_state::MCMCChainState,
    trafo_tuners::Vararg{MCMCTransformTunerState},
)
    _tune_proposal_components(multi_proposal, multi_tuner, chain_state) do proposal, tuner, chain
        component_chain = @set chain.proposal = proposal
        mcmc_proposal_transform_committed!!(proposal, tuner, component_chain, trafo_tuners...)
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
    _tune_proposal_components(mcmc_tune_proposal_post_cycle!!, multi_proposal, multi_tuner, chain_state, samples)
end


function mcmc_proposal_tuning_finalize!!(
    multi_proposal::MultiProposalState,
    multi_tuner::MultiProposalTunerState, 
    chain_state::MCMCChainState
)
    _tune_proposal_components(mcmc_proposal_tuning_finalize!!, multi_proposal, multi_tuner, chain_state)
end

function mcmc_tune_proposal_post_step!!(
    multi_proposal::MultiProposalState,
    multi_tuner::MultiProposalTunerState,
    chain_state::MCMCChainState,
    step_info::MCMCStepInfo
)
    _with_proposal_index(multi_proposal.proposal_states, multi_proposal.active_idx) do i
        _tune_proposal_component(mcmc_tune_proposal_post_step!!, multi_proposal, multi_tuner, chain_state, i, step_info)
    end
end

function _tune_proposal_component(f::F, multi_proposal, multi_tuner, chain_state, ::Val{I}, args...) where {F,I}
    proposal, tuner, chain = f(
        multi_proposal.proposal_states[I], multi_tuner.proposal_tuners[I], chain_state, args...,
    )
    proposals = @set multi_proposal.proposal_states[I] = proposal
    tuners = @set multi_tuner.proposal_tuners[I] = tuner
    chain = @set chain.proposal = proposals
    return proposals, tuners, chain
end

function _tune_proposal_components(f::F, multi_proposal, multi_tuner, chain_state, args...) where {F}
    indices = ntuple(Val, Val(length(multi_proposal.proposal_states)))
    foldl(indices; init = (multi_proposal, multi_tuner, chain_state)) do (proposals, tuners, chain), i
        _tune_proposal_component(f, proposals, tuners, chain, i, args...)
    end
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
