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
    scale::Float64 = 0.1
    base_picking_threshold::Float64 = 0.05
    beta::Float64 = 0.5
end

export AdaptiveMultiPropTuning

struct AdaptiveMultiPropTunerState <:MCMCProposalTunerState 
    scale::Float64
    base_picking_threshold::Float64
    beta::Float64
end

export AdaptiveMultiPropTunerState



function create_proposal_tuner_state(
    tuning::AdaptiveMultiPropTuning,
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
    iteration::Integer
)
    return AdaptiveMultiPropTunerState(
        tuning.scale,
        tuning.base_picking_threshold,
        tuning.beta
    )
end

mcmc_tuning_init!!(
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
) = nothing

mcmc_tuning_reinit!!(
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
) = nothing


mcmc_tuning_postinit!!(
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
) = nothing


function mcmc_tune_post_cycle!!(
    multi_proposal::MultiProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    # At the end of a cycle, multiply picking probabilities with a `tuning_quality_vector` that rewards proposals that 
    # lie in their respective target acceptance interval and punishes bad tuning. For this, add proposal-specific 
    # tuning quality methods. E.g. `IndependentMH` is never punished in this step.
    tuning_qualities = get_proposal_tuning_quality.(
        multi_proposal.proposal_states,
        Ref(chain_state),
        tuner_state.beta
    )
    picking_rule = multi_proposal.picking_rule
    base_thresh = tuner_state.base_picking_threshold

    global gs_tpc = (multi_proposal, tuner_state, chain_state, samples, tuning_qualities, picking_rule, base_thresh)
    # BREAK_TPC

    if picking_rule isa Distribution
        valid_proposals = picking_rule.p .> 0.0
        picking_probs_tuned = deepcopy(picking_rule.p)
        picking_probs_tuned .*= tuning_qualities

        if any(picking_probs_tuned .>0)
            picking_probs_tuned ./= sum(picking_probs_tuned)

            below_thresh = (picking_probs_tuned .< base_thresh) .* valid_proposals

            picking_probs_tuned[below_thresh] .= base_thresh
            rem_prob = 1.0 - sum(below_thresh) * base_thresh

            picking_probs_tuned[.!below_thresh] ./= (sum(picking_probs_tuned[.!below_thresh]) / rem_prob)
        else
            picking_probs_tuned[valid_proposals] .= 1/sum(valid_proposals)
        end

        picking_rule_tuned = Categorical(picking_probs_tuned)
    else
        valid_proposals = picking_rule .> 0.0
        N = sum(picking_rule)

        picking_rule_tuned = picking_rule ./ N
        picking_rule_tuned .*= tuning_qualities

        if any(picking_rule_tuned .> 0)
            picking_rule_tuned = round.(Integer, picking_rule_tuned .* (N / sum(picking_rule_tuned)))

            below_thresh = (picking_rule_tuned .< base_thresh) .* valid_proposals

            picking_rule_tuned[below_thresh] .= base_thresh
            rem_prob = 1.0 - sum(below_thresh) * (base_thresh/N)

            N_a = sum(picking_rule_tuned[.!below_thresh])

            picking_rule_tuned[.!below_thresh] ./= (sum(picking_probs_tuned[.!below_thresh]) / rem_prob)
            picking_rule_tuned[.!below_thresh] = round.(Integer, picking_rule_tuned[.!below_thresh] .* N_a)
        else
            picking_rule_tuned[valid_proposals] = sum(valid_proposals) / N
            picking_rule_tuned = round.(Integer, picking_rule_tuned .* N)
        end
    end

    multi_proposal_tuned = @set multi_proposal.picking_rule = picking_rule_tuned

    return multi_proposal_tuned, tuner_state, chain_state
end


function mcmc_tuning_finalize!!(
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
            picking_rule_tuned = round.(Integer, p_unnorm .* (N / sum(p_unnorm)))
        end
    else
        picking_rule_new = picking_rule
    end

    @reset multi_proposal.picking_rule = picking_rule_new

    return multi_proposal, tuner_state, chain_state
end

function mcmc_tune_post_step!!(
    multi_proposal::MultiProposalState,
    tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    p_accept::AbstractVector{<:Real}
)
    # Every time a proposal is picked, tune picking probability by
    # `picking_prob * acceptance_prob * small_delta`
    # and afterwards re-normalize picking probabilities
    # Ensure no picking probability falls below a base threshold.

    curr_idx = multi_proposal.current_idx
    picking_rule = multi_proposal.picking_rule
    n_proposals = length(multi_proposal.proposal_states)

    n_walkers = nwalkers(chain_state)
    scale = tuner_state.scale
    base_thresh = tuner_state.base_picking_threshold

    if picking_rule isa Distribution
        picking_probs_tuned = deepcopy(picking_rule.p)
        picking_probs_tuned[curr_idx] *= (1 + 1/n_walkers * sum(p_accept) * scale)
        picking_probs_tuned ./= sum(picking_probs_tuned)

        valid_proposal = picking_rule.p .> 0.0
        below_thresh = (picking_probs_tuned .< base_thresh) .* valid_proposal

        picking_probs_tuned[below_thresh] .= base_thresh
        rem_prob = 1.0 - sum(below_thresh) * base_thresh

        picking_probs_tuned[.!below_thresh] ./= (sum(picking_probs_tuned[.!below_thresh]) / rem_prob)
        picking_rule_tuned = Categorical(picking_probs_tuned)
    else
        N = sum(picking_rule)

        picking_rule_tuned = picking_rule ./ N
        picking_rule_tuned[curr_idx] *= (1 + 1/n_walkers * sum(p_accept) * scale)
        picking_rule_tuned = round.(Integer, picking_rule_tuned .* (N / sum(picking_rule_tuned)))

        valid_proposal = picking_rule .> 0.0
        below_thresh = (picking_rule_tuned .< base_thresh) .* valid_proposal

        picking_rule_tuned[below_thresh] .= base_thresh
        rem_prob = 1.0 - sum(below_thresh) * (base_thresh/N)

        N_a = sum(picking_rule_tuned[.!below_thresh])

        picking_rule_tuned[.!below_thresh] ./= (sum(picking_probs_tuned[.!below_thresh]) / rem_prob)
        picking_rule_tuned[.!below_thresh] = round.(Integer, picking_rule_tuned[.!below_thresh] .* N_a)
    end

    multi_proposal_tuned = @set multi_proposal.picking_rule = picking_rule_tuned

    return multi_proposal_tuned, tuner_state, chain_state
end

