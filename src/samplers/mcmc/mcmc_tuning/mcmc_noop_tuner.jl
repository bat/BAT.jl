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
get_tuner(tuning::MCMCNoOpTuning, chain::MCMCIterator) = MCMCNoOpTuner() 


function MCMCNoOpTuning(tuning::MCMCNoOpTuning, chain::MCMCIterator)
    MCMCNoOpTuner()
end


function tuning_init!(tuner::MCMCNoOpTuning, chain::MCMCIterator, max_nsteps::Integer)
    chain.info = MCMCIteratorInfo(chain.info, tuned = true)
    nothing
end



function tune_mcmc_transform!!(
    tuner::MCMCNoOpTuner, 
    transform,
    p_accept::Real,
    z_proposed::Vector{<:Float64}, #TODO: use DensitySamples instead
    z_current::Vector{<:Float64},
    stepno::Int,
    context::BATContext
)
    return (tuner, transform)

end

tuning_postinit!(tuner::MCMCNoOpTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_reinit!(tuner::MCMCNoOpTuner, chain::MCMCIterator, max_nsteps::Integer) = nothing

tuning_update!(tuner::MCMCNoOpTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

tuning_finalize!(tuner::MCMCNoOpTuner, chain::MCMCIterator) = nothing

tuning_callback(::MCMCNoOpTuning) = nop_func
