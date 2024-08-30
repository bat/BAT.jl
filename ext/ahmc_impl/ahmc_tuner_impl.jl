# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct AHMCTuner{A<:AdvancedHMC.AbstractAdaptor} <: AbstractMCMCTunerInstance
    target_acceptance::Float64
    adaptor::A
end

function (tuning::HMCTuningAlgorithm)(chain::MCMCState)
    θ = first(chain.samples).v
    adaptor = ahmc_adaptor(tuning, chain.hamiltonian.metric, chain.kernel.τ.integrator, θ)
    AHMCTuner(tuning.target_acceptance, adaptor)
end


function BAT.tuning_init!(tuner::AHMCTuner, chain::MCMCState, max_nsteps::Integer)
    AdvancedHMC.Adaptation.initialize!(tuner.adaptor, Int(max_nsteps - 1))
    nothing
end

BAT.tuning_postinit!(tuner::AHMCTuner, chain::MCMCState, samples::DensitySampleVector) = nothing

function BAT.tuning_reinit!(tuner::AHMCTuner, chain::MCMCState, max_nsteps::Integer)
    AdvancedHMC.Adaptation.initialize!(tuner.adaptor, Int(max_nsteps - 1))
    nothing
end

function BAT.tuning_update!(tuner::AHMCTuner, chain::MCMCState, samples::DensitySampleVector)
    max_log_posterior = maximum(samples.logd)
    accept_ratio = eff_acceptance_ratio(chain)
    if accept_ratio >= 0.9 * tuner.target_acceptance
        chain.info = MCMCStateInfo(chain.info, tuned = true)
        @debug "MCMC chain $(chain.info.id) tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain.proposal.τ.integrator), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain.info = MCMCStateInfo(chain.info, tuned = false)
        @debug "MCMC chain $(chain.info.id) *not* tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain.proposal.τ.integrator), max. log posterior = $(Float32(max_log_posterior))"
    end
    nothing
end

function BAT.tuning_finalize!(tuner::AHMCTuner, chain::MCMCState)
    adaptor = tuner.adaptor
    AdvancedHMC.finalize!(adaptor)
    chain.hamiltonian = AdvancedHMC.update(chain.hamiltonian, adaptor)
    chain.kernel = AdvancedHMC.update(chain.kernel, adaptor)
    nothing
end

BAT.tuning_callback(tuner::AHMCTuner) = AHMCTunerCallback(tuner)



struct AHMCTunerCallback{T<:AHMCTuner} <: Function
    tuner::T
end


function (callback::AHMCTunerCallback)(::Val{:mcmc_step}, chain::AHMCState)
    adaptor = callback.tuner.adaptor
    tstat = AdvancedHMC.stat(chain.transition)

    AdvancedHMC.adapt!(adaptor, chain.transition.z.θ, tstat.acceptance_rate)
    chain.hamiltonian = AdvancedHMC.update(chain.hamiltonian, adaptor)
    chain.kernel = AdvancedHMC.update(chain.kernel, adaptor)
    tstat = merge(tstat, (is_adapt =true,))

    nothing
end
