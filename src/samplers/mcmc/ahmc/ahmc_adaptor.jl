# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@with_kw struct MassMatrixAdaptor <: HMCTuningAlgorithm
    target_acceptance::Float64 = 0.8
end

function ahmc_adaptor(
    tuning::MassMatrixAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.MassMatrixAdaptor(metric)
end



@with_kw struct StepSizeAdaptor <: HMCTuningAlgorithm
    target_acceptance::Float64 = 0.8
end


function ahmc_adaptor(
    tuning::StepSizeAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.StepSizeAdaptor(tuning.target_acceptance, integrator)
end



@with_kw struct NaiveHMCTuning <: HMCTuningAlgorithm
    target_acceptance::Float64 = 0.8
end


function ahmc_adaptor(
    tuning::NaiveHMCTuning,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
)
    mma = AdvancedHMC.MassMatrixAdaptor(metric)
    ssa = AdvancedHMC.StepSizeAdaptor(tuning.target_acceptance, integrator)
    return AdvancedHMC.NaiveHMCTuning(mma, ssa)
end



# Uses Stan (also AdvancedHMC) defaults 
# (see https://mc-stan.org/docs/2_26/reference-manual/hmc-algorithm-parameters.html):
@with_kw struct StanHMCTuning <: HMCTuningAlgorithm
    "target acceptance rate"
    target_acceptance::Float64 = 0.8

    "width of initial fast adaptation interval"
    initial_bufsize::Int = 75

    "width of final fast adaptation interval"
    term_bufsize::Int = 50

    "initial width of slow adaptation interval"
    window_size::Int = 25
end


function ahmc_adaptor(
    tuning::StanHMCTuning,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator,
)
    mma = AdvancedHMC.MassMatrixAdaptor(metric)
    ssa = AdvancedHMC.StepSizeAdaptor(tuning.target_acceptance, integrator)
    stan_adaptor = AdvancedHMC.StanHMCAdaptor(
        mma, ssa,
        init_buffer = Int(tuning.initial_bufsize), term_buffer = Int(tuning.term_bufsize), window_size = Int(tuning.window_size)
    )
    return stan_adaptor
end



mutable struct AHMCTuner{A<:AdvancedHMC.AbstractAdaptor} <: AbstractMCMCTunerInstance
    target_acceptance::Float64
    adaptor::A
end

function (tuning::HMCTuningAlgorithm)(chain::MCMCIterator)
    adaptor = ahmc_adaptor(tuning, chain.hamiltonian.metric, chain.proposal.τ.integrator)
    AHMCTuner(tuning.target_acceptance, adaptor)
end


function tuning_init!(tuner::AHMCTuner, chain::MCMCIterator, max_nsteps::Integer)
    AdvancedHMC.Adaptation.initialize!(tuner.adaptor, Int(max_nsteps - 1))
    nothing
end

tuning_postinit!(tuner::AHMCTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

function tuning_reinit!(tuner::AHMCTuner, chain::MCMCIterator, max_nsteps::Integer)
    AdvancedHMC.Adaptation.initialize!(tuner.adaptor, Int(max_nsteps - 1))
    nothing
end

function tuning_update!(tuner::AHMCTuner, chain::MCMCIterator, samples::DensitySampleVector)
    max_log_posterior = maximum(samples.logd)
    accept_ratio = eff_acceptance_ratio(chain)
    if accept_ratio >= 0.9 * tuner.target_acceptance
        chain.info = MCMCIteratorInfo(chain.info, tuned = true)
        @debug "MCMC chain $(chain.info.id) tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain.proposal.integrator), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain.info = MCMCIteratorInfo(chain.info, tuned = false)
        @debug "MCMC chain $(chain.info.id) *not* tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain.proposal.integrator), max. log posterior = $(Float32(max_log_posterior))"
    end
    nothing
end

function tuning_finalize!(tuner::AHMCTuner, chain::MCMCIterator)
    adaptor = tuner.adaptor
    AdvancedHMC.finalize!(adaptor)
    chain.hamiltonian = AdvancedHMC.update(chain.hamiltonian, adaptor)
    chain.proposal = AdvancedHMC.update(chain.proposal, adaptor)
    nothing
end

tuning_callback(tuner::AHMCTuner) = AHMCTunerCallback(tuner)



struct AHMCTunerCallback{T<:AHMCTuner} <: Function
    tuner::T
end


function (callback::AHMCTunerCallback)(::Val{:mcmc_step}, chain::AHMCIterator)
    adaptor = callback.tuner.adaptor
    tstat = AdvancedHMC.stat(chain.transition)

    AdvancedHMC.adapt!(adaptor, chain.transition.z.θ, tstat.acceptance_rate)
    chain.hamiltonian = AdvancedHMC.update(chain.hamiltonian, adaptor)
    chain.proposal = AdvancedHMC.update(chain.proposal, adaptor)
    tstat = merge(tstat, (is_adapt =true,))

    nothing
end
