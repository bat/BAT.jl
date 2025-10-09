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
struct AdaptiveMultiPropTuning <:MCMCProposalTuning end

export AdaptiveMultiPropTuning

struct AdaptiveMultiPropTunerState <:MCMCProposalTunerState end

export AdaptiveMultiPropTunerState



function create_proposal_tuner_state(
    multi_tuning::AdaptiveMultiPropTuning,
    chain_state::MCMCChainState,
    multi_proposal::MultiProposalState,
    iteration::Integer
)
    return AdaptiveMultiPropTunerState()
end

mcmc_tuning_init!!(
    multi_tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
) = nothing

mcmc_tuning_reinit!!(
    multi_tuner_state::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
) = nothing


mcmc_tuning_postinit!!(
    multi_tuner::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
) = nothing


function mcmc_tune_post_cycle!!(
    multi_proposal::MultiProposalState,
    multi_tuner::AdaptiveMultiPropTunerState,
    chain_state::MCMCChainState,
    samples::AbstractVector{<:DensitySampleVector}
)
    proposals = multi_proposal.proposal_states
    target_acceptance_ratios = get_target_acceptance_ratio.(proposals)

    eff_acceptance_ratios = Tuple(detailed_eff_acceptance_ratio(chain_state))

    diff = target_acceptance_ratios .- eff_acceptance_ratios

    return multi_proposal, multi_tuner, chain_state
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

