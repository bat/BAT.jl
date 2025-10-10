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
@withkw struct AdaptiveMultiPropTuning <:MCMCProposalTuning 
    scale::Float64 = 1.5
end

export AdaptiveMultiPropTuning

struct AdaptiveMultiPropTunerState <:MCMCProposalTunerState 
    scale::Float64
end

export AdaptiveMultiPropTunerState



function create_proposal_tuner_state(
    tuning::AdaptiveMultiPropTuning,
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
    iteration::Integer
)
    return AdaptiveMultiPropTunerState(tuning.scale)
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
    proposals = multi_proposal.proposal_states
    target_acceptance_ratios = get_target_acceptance_ratio.(proposals)

    eff_acceptance_ratios = Tuple(detailed_eff_acceptance_ratio(chain_state))

    diff = abs.(target_acceptance_ratios .- eff_acceptance_ratios)

    picking_rule = multi_proposal.picking_rule

    scale = tuner_state.scale

    good_prop = diff .< 0.2

    if any(good_prop)
        if picking_rule isa Distribution
            picking_probs_tuned = picking_rule.p[good_prop] .* scale
            picking_probs_tuned ./= scale  
            picking_probs_tuned ./= sum(picking_probs_tuned)

            picking_rule_tuned = Categorical(picking_probs_tuned)
        else
            # How to adjust schedule-style picking rules?
            picking_rule_tuned = picking_rule
        end
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
    return multi_proposal, multi_tuner, chain_state
end

