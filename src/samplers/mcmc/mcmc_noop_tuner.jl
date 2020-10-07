# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Add literature references to AdaptiveMHTuning docstring.

"""
    MCMCNoOpTuning <: MCMCTuningAlgorithm

No-op tuning, leaves MCMC chains unmodified. Useful if chains are pre-tuned
or tuning is an internal part of the MCMC sampler implementation.
"""
struct MCMCNoOpTuning <: MCMCTuningAlgorithm end
export MCMCNoOpTuning


struct NoOpTuner end

(tuning::MCMCNoOpTuning)(chain::MCMCIterator) = NoOpTuner()

function MCMCNoOpTuning(tuning::MCMCNoOpTuning, chain::MCMCIterator)
    NoOpTuner()
end

tuning_init!(tuner::NoOpTuner, chain::MCMCIterator) = nothing

tuning_update!(tuner::NoOpTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing
