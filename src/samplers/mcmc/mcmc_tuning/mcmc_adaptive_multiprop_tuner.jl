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

struct AdaptiveMultiPropTunerState <:MCMCProposalTunerState 
    alpha::Float64
    beta::Float64
    picking_socket::Float64
    accept_prob::Vector{Float64} # initiate with 0.5
end

export AdaptiveMultiPropTunerState


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
    (; beta, picking_socket) = tuner_state
    
    tuning_qualities = get_proposal_tuning_quality.(
        proposal_states,
        Ref(chain_state),
        beta
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
    component_eff_acc= detailed_eff_acceptance_ratio(chain_state)
    component_acc_ints = get_target_acceptance_int.(multi_proposal.proposal_states)

    component_tuning_successes = [int[1] <= component_eff_acc[i] <= int[2] for (i, int) in enumerate(component_acc_ints)]

    picking_rule = multi_proposal.picking_rule

    if any(component_tuning_successes)
        if picking_rule isa Distribution
            p_unnorm = picking_rule.p .* component_tuning_successes
            picking_probs_new = p_unnorm ./ sum(p_unnorm)
            picking_rule_new = Categorical(picking_probs_new)
        else
            N = sum(picking_rule)
            p_unnorm = picking_rule .* component_tuning_successes
            picking_rule_new = round.(Integer, p_unnorm .* (N / sum(p_unnorm)))
        end
    else
        picking_rule_new = picking_rule
    end

    @reset multi_proposal.picking_rule = picking_rule_new

    return multi_proposal, tuner_state, chain_state
end

function mcmc_tune_proposal_post_step!!(
    multi_proposal::MultiProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    p_accept::AbstractVector{<:Real}
)
    (;alpha, picking_socket, accept_prob) = tuner_state
    active_idx = multi_proposal.active_idx
    picking_rule = multi_proposal.picking_rule
    N = length(multi_proposal.proposal_states)
 
    acc_new = accept_prob[active_idx] * (1-alpha) + mean(p_accept) * alpha
    accept_prob[active_idx] = acc_new

    picking_rule_tuned = _tune_picking_rule(picking_rule, acc_new, curr_idx, picking_socket, N)

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
    p_tuned = picking_rule.p
    p_tuned[curr_idx] = acc_new
    p_tuned .*= (1 - picking_socket) / sum(p_tuned)
    p_tuned .+= picking_socket / N  
    return Categorical(p_tuned)
end

function _tune_picking_rule(
    picking_rule::Tuple,
    acc_new::Float64,
    curr_idx::Integer,
    picking_socket::Float64,
    N::Integer
)
    norm = sum(picking_rule)
    picking_rule_tuned = picking_rule ./ norm
    picking_rule_tuned[curr_idx] = acc_new
    picking_rule_tuned .*=  (1 - picking_socket) / sum(picking_rule_tuned)
    picking_rule_tuned .+= picking_socket / N

    return round.(Integer, picking_rule_tuned * norm)
end

function _qualify_picking_rule(
    picking_rule::Categorical,
    tuning_qualities::Tuple,
    picking_socket::Float64,
    N_props::Integer
)
    valid_proposals = picking_rule.p .> 0.0
    @assert any(valid_proposals) throw Error("All proposals have picking probability 0!")

    p_tuned = picking_rule.p
    p_tuned .*= tuning_qualities

    if any(p_tuned .> 0)
        p_tuned ./= sum(p_tuned)

        p_tuned .*= (1 - picking_socket) / sum(p_tuned)
        p_tuned .+= picking_socket / N_props
    else
        p_tuned[valid_proposals] .= 1/sum(valid_proposals)
        @warn "No proposal was tuned to its target acceptance interval."        
    end

    return Categorical(p_tuned)
end

function _qualify_picking_rule(
    picking_rule::Tuple,
    tuning_qualities::Tuple,
    picking_socket::Float64,
    N_props::Integer
)
    N = sum(picking_rule)
    valid_proposals = picking_rule .> 0 
    @assert any(valid_proposals) throw Error("All proposals have picking probability 0!")

    picking_rule_tuned = picking_rule ./ N
    picking_rule_tuned .*= tuning_qualities

    if any(picking_rule_tuned .> 0)
        picking_rule_tuned .*= (1 - picking_socket) / sum(p_tuned)
        picking_rule_tuned .+= picking_socket / N_props
       
        picking_rule_tuned = round.(Integer, picking_rule_tuned .* N)
    else
        picking_rule_tuned[valid_proposals] = sum(valid_proposals) / N
        picking_rule_tuned = round.(Integer, picking_rule_tuned .* N)
    end

    return picking_rule_tuned 
end