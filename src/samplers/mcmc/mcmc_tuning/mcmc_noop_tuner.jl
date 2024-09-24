# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    MCMCNoOpTuning <: MCMCTuning

No-op tuning, marks MCMC chain states as tuned without performing any other changes
on them. Useful if chain states are pre-tuned or tuning is an internal part of the
MCMC sampler implementation.
"""
struct MCMCNoOpTuning <: MCMCTuning end
export MCMCNoOpTuning



struct MCMCNoOpTuner <: AbstractMCMCTunerInstance end

(tuning::MCMCNoOpTuning)(mc_state::MCMCState) = MCMCNoOpTuner()


function MCMCNoOpTuning(tuning::MCMCNoOpTuning, mc_state::MCMCState)
    MCMCNoOpTuner()
end


function tuning_init!(tuner::MCMCNoOpTuning, mc_state::MCMCState, max_nsteps::Integer)
    mc_state.info = MCMCStateInfo(mc_state.info, tuned = true)
    nothing
end


tuning_postinit!(tuner::MCMCNoOpTuner, mc_state::MCMCState, samples::DensitySampleVector) = nothing

tuning_reinit!(tuner::MCMCNoOpTuner, mc_state::MCMCState, max_nsteps::Integer) = nothing

tuning_update!(tuner::MCMCNoOpTuner, mc_state::MCMCState, samples::DensitySampleVector) = nothing

tuning_finalize!(tuner::MCMCNoOpTuner, mc_state::MCMCState) = nothing

tuning_callback(::MCMCNoOpTuning) = nop_func
