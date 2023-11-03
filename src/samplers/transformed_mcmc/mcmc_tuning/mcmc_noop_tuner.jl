# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    TransformedMCMCNoOpTuning <: TransformedMCMCTuningAlgorithm

No-op tuning, marks MCMC chains as tuned without performing any other changes
on them. Useful if chains are pre-tuned or tuning is an internal part of the
MCMC sampler implementation.
"""
struct TransformedMCMCNoOpTuning <: TransformedMCMCTuningAlgorithm end
export TransformedMCMCNoOpTuning



struct TransformedMCMCNoOpTuner <: TransformedAbstractMCMCTunerInstance end

(tuning::TransformedMCMCNoOpTuning)(chain::MCMCIterator) = TransformedMCMCNoOpTuner()
get_tuner(tuning::TransformedMCMCNoOpTuning, chain::MCMCIterator) = TransformedMCMCNoOpTuner() 


function TransformedMCMCNoOpTuning(tuning::TransformedMCMCNoOpTuning, chain::MCMCIterator)
    TransformedMCMCNoOpTuner()
end


function tuning_init!(tuner::TransformedMCMCNoOpTuning, chain::MCMCIterator, max_nsteps::Integer)
    chain.info = TransformedMCMCIteratorInfo(chain.info, tuned = true)
    nothing
end



function tune_mcmc_transform!!(
    tuner::TransformedMCMCNoOpTuner, 
    transform,
    p_accept::Real,
    z_proposed::Vector{<:Float64}, #TODO: use DensitySamples instead
    z_current::Vector{<:Float64},
    stepno::Int,
    context::BATContext
)
    return (tuner, transform)

end

tuning_postinit!(tuner::TransformedMCMCNoOpTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_reinit!(tuner::TransformedMCMCNoOpTuner, chain::MCMCIterator, max_nsteps::Integer) = nothing

tuning_update!(tuner::TransformedMCMCNoOpTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_finalize!(tuner::TransformedMCMCNoOpTuner, chain::MCMCIterator) = nothing

tuning_callback(::TransformedMCMCNoOpTuning) = nop_func
