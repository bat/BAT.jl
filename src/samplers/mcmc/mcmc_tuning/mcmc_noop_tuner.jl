# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    NoMCMCTransformTuning <: MCMCTransformTuning

Do not perform any MCMC transform turing.
"""
struct NoMCMCTransformTuning <: MCMCTransformTuning end
export NoMCMCTransformTuning

struct NoMCMCTransformTuningState <: MCMCTransformTunerState end


create_trafo_tuner_state(
    ::NoMCMCTransformTuning,
    ::MCMCChainState,
    ::Integer
) = NoMCMCTransformTuningState()

mcmc_trafo_tuning_init!!(
    ::NoMCMCTransformTuningState,
    ::MCMCChainState,
    ::Integer
) = nothing

mcmc_trafo_tuning_reinit!!(
    ::NoMCMCTransformTuningState,
    ::MCMCChainState,
    ::Integer
) = nothing

mcmc_trafo_tuning_postinit!!(
    ::NoMCMCTransformTuningState,
    ::MCMCChainState,
    ::AbstractVector{<:DensitySampleVector}
) = nothing

mcmc_tune_trafo_post_cycle!!(
    f_transform::Function,
    tuner::NoMCMCTransformTuningState,
    chain_state::MCMCChainState,
    ::AbstractVector{<:DensitySampleVector}
) = f_transform, tuner, chain_state

mcmc_trafo_tuning_finalize!!(
    f_transform::Function,
    trafo_tuner_state::NoMCMCTransformTuningState,
    chain_state::MCMCChainState
) = f_transfrom, trafo_tuner_state, chain_state

mcmc_tune_trafo_post_step!!(f_transform::Function, tuner::NoMCMCTransformTuningState, chain_state::MCMCChainState, ::NamedTuple, ::NamedTuple, ::AbstractVector{<:Real}) = f_transform, tuner, chain_state



"""
    NoMCMCProposalTuning <: MCMCProposalTuning

Do not perform any MCMC proposal tuning.
"""
struct NoMCMCProposalTuning <: MCMCProposalTuning end
export NoMCMCProposalTuning

struct NoMCMCProposalTunerState <: MCMCProposalTunerState end


create_proposal_tuner_state(
    ::NoMCMCProposalTuning,
    ::MCMCChainState,
    ::MCMCProposalState,
    ::Integer
) = NoMCMCProposalTunerState()

mcmc_proposal_tuning_init!!(
    ::NoMCMCProposalTunerState,
    ::MCMCChainState,
    ::Integer
) = nothing

mcmc_proposal_tuning_reinit!!(
    ::NoMCMCProposalTunerState,
    ::MCMCChainState,
    ::Integer
) = nothing

mcmc_proposal_tuning_postinit!!(
    ::NoMCMCProposalTunerState,
    ::MCMCChainState,
    ::AbstractVector{<:DensitySampleVector}
) = nothing

mcmc_tune_proposal_post_cycle!!(
    proposal::MCMCProposalState, 
    tuner::NoMCMCProposalTunerState, 
    chain_state::MCMCChainState, 
    ::AbstractVector{<:DensitySampleVector}
) = proposal, tuner, chain_state

mcmc_proposal_tuning_finalize!!(
    proposal_state::MCMCProposalState,
    proposal_tuner_state::NoMCMCProposalTunerState, 
    chain_state::MCMCChainState
) = proposal_state, proposal_tuner_state, chain_state

mcmc_tune_proposal_post_step!!(
    proposal::MCMCProposalState, 
    tuner::NoMCMCProposalTunerState, 
    chain_state::MCMCChainState, 
    ::AbstractVector{<:Real}
) = proposal, tuner, chain_state 
