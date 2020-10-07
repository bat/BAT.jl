# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Add literature references to AdaptiveMHTuning docstring.

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


function tuning_init!(tuner::MCMCNoOpTuner, chain::MCMCIterator)
    chain.info = MCMCIteratorInfo(chain.info, cycle = chain.info.cycle + 1)
    nothing
end


function tuning_update!(tuner::MCMCNoOpTuner, chain::MCMCIterator, samples::DensitySampleVector)
    chain.info = MCMCIteratorInfo(chain.info, tuned = true)
    nothing
end
