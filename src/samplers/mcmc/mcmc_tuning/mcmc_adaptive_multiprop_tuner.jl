# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct AdaptiveMultiPropTuning <: MCMCProposalTuning

Tuning Algorithm for multiple MCMC Proposals. Works by adjusting the picking
rule for the proposals to match the individual desired target acceptance rates
based on the respective observed acceptance rates.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct AdaptiveMultiPropTuning <:MCMCProposalTuning 
    alpha::Float64 = 0.1
    beta::Float64 = 0.5
    picking_socket::Float64 = 0.8
end

export AdaptiveMultiPropTuning

_validate_mcmc_proposal_configuration(
    proposal::Union{RandomWalk,MCMCGlobalProposal,MALAProposal,HamiltonianMC},
    tuning::Union{MultiProposalTuning,AdaptiveMultiPropTuning},
) = _unsupported_mcmc_component_tuning(proposal, tuning)

_validate_mcmc_proposal_configuration(
    proposal::Union{RandomWalk,MCMCGlobalProposal},
    tuning::HMCTuning,
) = _unsupported_mcmc_component_tuning(proposal, tuning)

function _validate_mcmc_proposal_tuning_configuration(
    multi_proposal::MCMCMultiProposal,
    ::AdaptiveMultiPropTuning,
)
    multi_proposal.picking_rule isa Categorical || throw(ArgumentError(
        "AdaptiveMultiPropTuning supports only categorical MCMCMultiProposal picking rules",
    ))
    return nothing
end

struct AdaptiveMultiPropTunerState <:MCMCProposalTunerState 
    alpha::Float64
    beta::Float64
    picking_socket::Float64
    accept_prob::Vector{Float64} # initiate with 0.5
end



function create_proposal_tuner_state(
    tuning::AdaptiveMultiPropTuning,
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
    iteration::Integer
)
    N_proposals = length(multi_proposal.proposal_states)

    return AdaptiveMultiPropTunerState(
        tuning.alpha,
        tuning.beta,
        tuning.picking_socket,
        fill(0.5, N_proposals)
    )
end

mcmc_proposal_tuning_init!!(
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
) = nothing

mcmc_proposal_tuning_reinit!!(
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
) = nothing


mcmc_proposal_tuning_postinit!!(
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
) = nothing


function mcmc_tune_proposal_post_cycle!!(
    multi_proposal::MultiProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    (; proposal_states, picking_rule) = multi_proposal
    (; picking_socket) = tuner_state

    tuning_qualities = _adaptive_component_tuning_qualities(
        proposal_states, tuner_state, chain_state,
    )

    picking_rule_tuned = _qualify_picking_rule(
        picking_rule,
        tuning_qualities,
        picking_socket,
        length(proposal_states)
    )

    multi_proposal_tuned = @set multi_proposal.picking_rule = picking_rule_tuned

    return multi_proposal_tuned, tuner_state, chain_state
end


function mcmc_proposal_tuning_finalize!!(
    multi_proposal::MultiProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState
)
    component_tuning_successes = _adaptive_component_tuning_successes(
        multi_proposal.proposal_states, tuner_state, chain_state,
    )

    picking_rule = multi_proposal.picking_rule

    picking_rule_new = if any(component_tuning_successes)
        p_unnorm = picking_rule.p .* component_tuning_successes
        Categorical(p_unnorm ./ sum(p_unnorm))
    else
        picking_rule
    end

    @reset multi_proposal.picking_rule = picking_rule_new

    return multi_proposal, tuner_state, chain_state
end

get_tuning_success(
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
) = any(_adaptive_component_tuning_successes(
    multi_proposal.proposal_states, tuner_state, chain_state,
))

function _adaptive_component_tuning_qualities(
    proposal_states,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
)
    return map(eachindex(proposal_states)) do i
        chain_state.nattempts[i] > 0 || return 0.0
        _adaptive_component_tuning_quality(
            proposal_states[i], tuner_state, chain_state, i,
        )
    end
end

function _adaptive_component_tuning_successes(
    proposal_states,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
)
    return map(eachindex(proposal_states)) do i
        chain_state.nattempts[i] > 0 || return false
        _adaptive_component_tuning_success(
            proposal_states[i], tuner_state, chain_state, i,
        )
    end
end

_adaptive_component_tuning_success(
    proposal::MCMCProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    i::Integer,
) = _adaptive_component_tuning_quality(
    proposal, tuner_state, chain_state, i,
) > 0

_adaptive_component_tuning_success(
    proposal::HMCProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    ::MCMCChainState,
    i::Integer,
) = _acceptance_in_target(proposal, tuner_state.accept_prob[i])

_adaptive_component_tuning_success(
    proposal::SimpleMCMCProposalState,
    ::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    i::Integer,
) = _acceptance_in_target(
    proposal, chain_state.nsamples[i] / chain_state.nattempts[i],
)

_adaptive_component_tuning_quality(
    proposal::MCMCProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    ::Integer,
) = get_proposal_tuning_quality(proposal, chain_state, tuner_state.beta)

_adaptive_component_tuning_quality(
    proposal::HMCProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    ::MCMCChainState,
    i::Integer,
) = get_proposal_tuning_quality(
    proposal, tuner_state.accept_prob[i], tuner_state.beta,
)

_adaptive_component_tuning_quality(
    proposal::SimpleMCMCProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    i::Integer,
) = get_proposal_tuning_quality(
    proposal, chain_state.nsamples[i] / chain_state.nattempts[i], tuner_state.beta,
)

function mcmc_tune_proposal_post_step!!(
    multi_proposal::MultiProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    step_info::MCMCStepInfo
)
    p_accept = step_info.p_accept
    (;alpha, picking_socket, accept_prob) = tuner_state
    active_idx = multi_proposal.active_idx
    picking_rule = multi_proposal.picking_rule
    N = length(multi_proposal.proposal_states)
 
    accept_sum = _ordered_walker_sum(p_accept, step_info.walker_order)
    mean_accept = accept_sum / length(p_accept)
    acc_new = accept_prob[active_idx] * (1-alpha) + mean_accept * alpha
    accept_prob[active_idx] = acc_new

    picking_rule_tuned = _tune_picking_rule(picking_rule, acc_new, active_idx, picking_socket, N)

    multi_proposal_tuned = @set multi_proposal.picking_rule = picking_rule_tuned

    return multi_proposal_tuned, tuner_state, chain_state
end

function _tune_picking_rule(
    picking_rule::Categorical,
    acc_new::Float64,
    curr_idx::Integer,
    picking_socket::Float64,
    N::Integer
)
    p_tuned = copy(picking_rule.p)
    p_tuned[curr_idx] = acc_new
    total = sum(p_tuned)
    if iszero(total)
        fill!(p_tuned, 1 / N)
    else
        p_tuned .*= (1 - picking_socket) / total
        p_tuned .+= picking_socket / N
    end
    return Categorical(p_tuned)
end

function _qualify_picking_rule(
    picking_rule::Categorical,
    tuning_qualities::AbstractVector{<:Real},
    picking_socket::Float64,
    N_props::Integer
)
    valid_proposals = picking_rule.p .> 0.0
    @assert any(valid_proposals) throw Error("All proposals have picking probability 0!")

    p_tuned = copy(picking_rule.p)
    p_tuned .*= tuning_qualities

    if any(p_tuned .> 0)
        p_tuned .*= (1 - picking_socket) / sum(p_tuned)
        p_tuned .+= picking_socket / N_props
    else
        p_tuned[valid_proposals] .= 1/sum(valid_proposals)
        @warn "No proposal was tuned to its target acceptance interval."        
    end

    return Categorical(p_tuned)
end
