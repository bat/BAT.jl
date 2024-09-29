# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    MCMCNoOpTuning <: MCMCTuning

No-op tuning, marks MCMC chain states as tuned without performing any other changes
on them. Useful if chain states are pre-tuned or tuning is an internal part of the
MCMC sampler implementation.
"""
struct MCMCNoOpTuning <: MCMCTuning end
export MCMCNoOpTuning

struct MCMCNoOpTunerState <: MCMCTunerState end

(tuning::MCMCNoOpTuning)(mc_state::MCMCChainState) = MCMCNoOpTunerState(), MCMCNoOpTunerState()

default_adaptive_transform(tuning::MCMCNoOpTuning) = nop_func

function NoOpTunerState(tuning::MCMCNoOpTuning, mc_state::MCMCChainState, iteration::Integer)
    MCMCNoOpTunerState()
end

create_trafo_tuner_state(tuning::MCMCNoOpTuning, mc_state::MCMCChainState, iteration::Integer) = MCMCNoOpTunerState()

create_proposal_tuner_state(tuning::MCMCNoOpTuning, mc_state::MCMCChainState, iteration::Integer) = MCMCNoOpTunerState()

mcmc_tuning_init!!(tuner_state::MCMCNoOpTunerState, mc_state::MCMCChainState, max_nsteps::Integer) = nothing

mcmc_tuning_reinit!!(tuner::MCMCNoOpTunerState, mc_state::MCMCChainState, max_nsteps::Integer) = nothing

mcmc_tuning_postinit!!(tuner::MCMCNoOpTunerState, mc_state::MCMCChainState, samples::DensitySampleVector) = nothing

mcmc_tune_post_cycle!!(tuner::MCMCNoOpTunerState, mc_state::MCMCChainState, samples::DensitySampleVector) = nothing

mcmc_tuning_finalize!!(tuner::MCMCNoOpTunerState, mc_state::MCMCChainState) = nothing

tuning_callback(::MCMCNoOpTuning) = nop_func

tuning_callback(::Nothing) = nop_func


function mcmc_tune_post_step!!(chain_state::MCMCChainState, tuner::MCMCNoOpTunerState, ::Real)
    return chain_state, tuner, false
end

function mcmc_tune_post_step!!(chain_state::MCMCChainState, tuner::Nothing, ::Real)
    return chain_state, nothing, false
end
