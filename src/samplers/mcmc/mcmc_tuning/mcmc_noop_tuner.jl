# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    MCMCNoOpTuning <: MCMCTuning

No-op tuning, marks MCMC chain states as tuned without performing any other changes
on them. Useful if chain states are pre-tuned or tuning is an internal part of the
MCMC sampler implementation.
"""
struct MCMCNoOpTuning <: MCMCTuning end
export MCMCNoOpTuning



struct MCMCNoOpTunerState <: AbstractMCMCTunerInstance end

(tuning::MCMCNoOpTuning)(mc_state::MCMCState) = MCMCNoOpTunerState()


function MCMCNoOpTuning(tuning::MCMCNoOpTuning, mc_state::MCMCState)
    MCMCNoOpTunerState()
end


function tuning_init!(tuner::MCMCNoOpTuning, mc_state::MCMCState, max_nsteps::Integer)
    mc_state.info = MCMCStateInfo(mc_state.info, tuned = true)
    nothing
end


tuning_postinit!(tuner::MCMCNoOpTunerState, mc_state::MCMCState, samples::DensitySampleVector) = nothing

tuning_reinit!(tuner::MCMCNoOpTunerState, mc_state::MCMCState, max_nsteps::Integer) = nothing

tuning_update!(tuner::MCMCNoOpTunerState, mc_state::MCMCState, samples::DensitySampleVector) = nothing

tuning_finalize!(tuner::MCMCNoOpTunerState, mc_state::MCMCState) = nothing

tuning_callback(::MCMCNoOpTuning) = nop_func

function mcmc_tune_transform!!(mc_state::MCMCState, tuner::MCMCNoOpTunerState, ::Real)
    return (tuner, mc_state.f_transform)
end

function mcmc_tune_transform!!(mc_state::MCMCState, tuner::Nothing, ::Real)
    return (nothing, mc_state.f_transform)
end
