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
    stepno::Int
    n_adapts::Int
end

function (tuning::HMCTuningAlgorithm)(chain::MCMCIterator)
    adaptor = ahmc_adaptor(tuning, chain.hamiltonian.metric, chain.proposal.integrator)
    AHMCTuner(tuning.target_acceptance, adaptor, 0, 0)
end


function tuning_init!(tuner::AHMCTuner, chain::MCMCIterator, max_nsteps::Int)
    tuner.stepno = 0
    tuner.n_adapts = 0
    nothing
end


tuning_postinit!(tuner::AHMCTuner, chain::MCMCIterator, samples::DensitySampleVector) = nothing

function tuning_reinit!(tuner::AHMCTuner, chain::MCMCIterator, max_nsteps::Int)
    tuner.stepno = 0
    tuner.n_adapts = max_nsteps - 1
end

function tuning_update!(tuner::AHMCTuner, chain::MCMCIterator, samples::DensitySampleVector)
    max_log_posterior = maximum(samples.logd)
    accept_ratio = eff_acceptance_ratio(chain)
    if accept_ratio >= tuner.target_acceptance
        chain.info = MCMCIteratorInfo(chain.info, tuned = true)
        @debug "MCMC chain $(chain.info.id) tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain.proposal.integrator), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain.info = MCMCIteratorInfo(chain.info, tuned = false)
        @debug "MCMC chain $(chain.info.id) *not* tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain.proposal.integrator), max. log posterior = $(Float32(max_log_posterior))"
    end
end


tuning_callback(tuner::AHMCTuner) = AHMCTunerCallback(tuner)




struct AHMCTunerCallback{T<:AHMCTuner} <: Function
    tuner::T
end


function (callback::AHMCTunerCallback)(::Val{:mcmc_step}, chain::AHMCIterator)
    tuner = callback.tuner
    adaptor = tuner.adaptor
    n_adapts = tuner.n_adapts

    if n_adapts > 0
        tuner.stepno += 1

        i = tuner.stepno
        # First value of i must be 1 for AdvancedHMC.adapt! to initialize
        # chain.adaptor, so i must never be less than 1:
        @assert i >= 1

        tstat = AdvancedHMC.stat(chain.transition)

        # First value of i must be 1 for AdvancedHMC.adapt! to initialize chain.adaptor!
        chain.hamiltonian, chain.proposal, isadapted = AdvancedHMC.adapt!(
            chain.hamiltonian,
            chain.proposal,
            adaptor,
            Int(i),
            Int(n_adapts),
            chain.transition.z.θ,
            tstat.acceptance_rate
        )

        tstat = merge(tstat, (is_adapt=isadapted,))
    end
end
