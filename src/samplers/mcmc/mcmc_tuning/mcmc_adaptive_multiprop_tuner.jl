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
    scale::Float64 = 1.5
    base_picking_threshold::Float64 = 0.05
end

export AdaptiveMultiPropTuning

struct AdaptiveMultiPropTunerState <:MCMCProposalTunerState 
    scale::Float64
    base_picking_threshold::Float64
end

export AdaptiveMultiPropTunerState



function create_proposal_tuner_state(
    tuning::AdaptiveMultiPropTuning,
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
    iteration::Integer
)
    return AdaptiveMultiPropTunerState(tuning.scale, tuning.base_picking_threshold)
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
    # tuning quality methods. E.g. `GlobalProposal` is never punished in this step.
    tuning_qualities = get_proposal_tuning_quality.(multi_proposal.proposal_states)
    if picking_rule isa Distribution
        picking_probs_tuned = picking_rule.p .* tuning_qualities
        picking_probs_tuned ./= sum(picking_probs_tuned)
        picking_rule_tuned = Categorical(picking_probs_tuned)
    else
        N = sum(picking_rule)
        picking_rule_tuned = picking_rule ./ N .* tuning_qualities
        picking_rule_tuned = round.(Integer, picking_rule_tuned .* (N / sum(picking_rule_tuned))
    end

    multi_proposal_tuned = @set multi_proposal.picking_rule = picking_rule_tuned

    return multi_proposal_tuned, multi_tuner, chain_state
end


mcmc_tuning_finalize!!(
    multi_proposal::MultiProposalState,
    multi_tuner::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState
) = nothing

function mcmc_tune_post_step!!(
    multi_proposal::MultiProposalState,
    multi_tuner::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    p_accept::AbstractVector{<:Real}
)
    # Every time a proposal is picked, tune picking probability by
    # `picking_prob * acceptance_prob * small_delta`
    # and afterwards re-normalize picking probabilities

    curr_idx = multiproposal.current_idx
    picking_rule = multi_proposal.picking_rule
    n_proposals = lenght(multi_proposal.proposals)

    n_walkers = nwalkers(chain_state)
    scale = multi_tuner.scale

    if picking_rule isa Distribution
        picking_probs_tuned = picking_rule.p

        picking_probs_tuned[curr_idx] *= (1 + 1/n_walkers * sum(p_accept) * scale)

        picking_probs_tuned ./= sum(picking_probs_tuned)

        picking_rule_tuned = Categorical(picking_probs_tuned)
    else
        N = sum(picking_rule)
        picking_rule_tuned = picking_rule ./ N

        picking_rule_tuned[curr_idx] *= (1 + 1/n_walkers * sum(p_accept) * scale)

        picking_rule_tuned = round.(Integer, picking_rule_tuned .* (N / sum(picking_rule_tuned))
    end

    multi_proposal_tuned = @set multi_proposal.picking_rule = picking_rule_tuned

    return multi_proposal_tuned, multi_tuner, chain_state
end

