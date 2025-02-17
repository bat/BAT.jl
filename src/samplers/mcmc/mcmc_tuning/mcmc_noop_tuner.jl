# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    NoMCMCTransformTuning <: MCMCTransformTuning

Do not perform any MCMC transform turing.
"""
struct NoMCMCTransformTuning <: MCMCTransformTuning end
export NoMCMCTransformTuning

struct NoMCMCTransformTuningState <: MCMCTransformTunerState end


create_trafo_tuner_state(::NoMCMCTransformTuning, ::MCMCChainState, ::Integer) = NoMCMCTransformTuningState()

mcmc_tuning_init!!(::NoMCMCTransformTuningState, ::MCMCChainState, ::Integer) = nothing

mcmc_tuning_reinit!!(::NoMCMCTransformTuningState, ::MCMCChainState, ::Integer) = nothing

mcmc_tuning_postinit!!(::NoMCMCTransformTuningState, ::MCMCChainState, ::DensitySampleVector) = nothing

mcmc_tune_post_cycle!!(tuner::NoMCMCTransformTuningState, chain_state::MCMCChainState, ::DensitySampleVector) = chain_state, tuner, false

mcmc_tuning_finalize!!(::NoMCMCTransformTuningState, ::MCMCChainState) = nothing

mcmc_tune_post_step!!(tuner::NoMCMCTransformTuningState, chain_state::MCMCChainState, ::Real) = chain_state, tuner, false



"""
    NoMCMCProposalTuning <: MCMCProposalTuning

Do not perform any MCMC proposal tuning.
"""
struct NoMCMCProposalTuning <: MCMCProposalTuning end
export NoMCMCProposalTuning

struct NoMCMCProposalTunerState <: MCMCProposalTunerState end


create_proposal_tuner_state(::NoMCMCProposalTuning, ::MCMCChainState, ::Integer) = NoMCMCProposalTunerState()

mcmc_tuning_init!!(::NoMCMCProposalTunerState, ::MCMCChainState, ::Integer) = nothing

mcmc_tuning_reinit!!(::NoMCMCProposalTunerState, ::MCMCChainState, ::Integer) = nothing

mcmc_tuning_postinit!!(::NoMCMCProposalTunerState, ::MCMCChainState, ::DensitySampleVector) = nothing

mcmc_tune_post_cycle!!(tuner::NoMCMCProposalTunerState, chain_state::MCMCChainState, ::DensitySampleVector) = chain_state, tuner

mcmc_tuning_finalize!!(::NoMCMCProposalTunerState, ::MCMCChainState) = nothing

mcmc_tune_post_step!!(tuner::NoMCMCProposalTunerState, chain_state::MCMCChainState, ::Real) = chain_state, tuner
