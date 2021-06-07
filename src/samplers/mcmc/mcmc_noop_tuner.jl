# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    MCMCNoOpTuning <: MCMCTuningAlgorithm

No-op tuning, marks MCMC chains as tuned without performing any other changes
on them. Useful if chains are pre-tuned or tuning is an internal part of the
MCMC sampler implementation.
"""
struct MCMCNoOpTuning <: MCMCTuningAlgorithm end
export MCMCNoOpTuning



struct MCMCNoOpTuner <: AbstractMCMCTunerInstance end

(tuning::MCMCNoOpTuning)(chain::MCMCIterator) = MCMCNoOpTuner()


function MCMCNoOpTuning(tuning::MCMCNoOpTuning, chain::MCMCIterator)
    MCMCNoOpTuner()
end


function tuning_init!(tuner::MCMCNoOpTuning, chain::MCMCIterator, max_nsteps::Int)
    chain.info = MCMCIteratorInfo(chain.info, tuned = true)
    nothing
end


tuning_postinit!(tuner::MCMCNoOpTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_reinit!(tuner::MCMCNoOpTuner, chain::MCMCIterator, max_nsteps::Int) = nothing

tuning_update!(tuner::MCMCNoOpTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_callback(::MCMCNoOpTuning) = nop_func
