# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Integrator ==============================================

function _ahmc_set_step_size(integrator::AdvancedHMC.AbstractIntegrator, hamiltonian::AdvancedHMC.Hamiltonian, θ_init::AbstractVector{<:Real}, rng::AbstractRNG)
    # ToDo: Add way to specify max_n_iters
    T = eltype(θ_init)
    step_size = integrator.ϵ
    if isnan(step_size)
        new_step_size = AdvancedHMC.find_good_stepsize(rng, hamiltonian, θ_init, max_n_iters = 100)
        @set integrator.ϵ = T(new_step_size)
    else
        @set integrator.ϵ = T(step_size)
    end
end


# Metric ==============================================

function ahmc_metric(metric::DiagEuclideanMetric, θ_init::AbstractVector{<:Real})
    return AdvancedHMC.DiagEuclideanMetric(eltype(θ_init), size(θ_init, 1))
end

function ahmc_metric(metric::UnitEuclideanMetric, θ_init::AbstractVector{<:Real})
    return AdvancedHMC.UnitEuclideanMetric(eltype(θ_init), size(θ_init, 1))
end

function ahmc_metric(metric::DenseEuclideanMetric, θ_init::AbstractVector{<:Real})
    return AdvancedHMC.DenseEuclideanMetric(eltype(θ_init), size(θ_init, 1))
end

# Termination =========================================

function _ahmc_convert_termination(termination::AdvancedHMC.AbstractTerminationCriterion, θ_init::AbstractVector{<:Real})
    T = eltype(θ_init)
    @set termination.Δ_max = T(termination.Δ_max)
end


# Tuning ==============================================


function ahmc_adaptor(
    tuning::MassMatrixAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator,
    θ_init::AbstractVector{<:Real},
    target_acceptance::Real
)
    return AdvancedHMC.MassMatrixAdaptor(metric)
end

function ahmc_adaptor(
    tuning::StepSizeAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator,
    θ_init::AbstractVector{<:Real},
    target_acceptance::Real
)
    T = eltype(θ_init)
    return AdvancedHMC.StepSizeAdaptor(T(target_acceptance), integrator)
end

function ahmc_adaptor(
    tuning::NaiveHMCTuning,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator,
    θ_init::AbstractVector{<:Real},
    target_acceptance::Real
)
    T = eltype(θ_init)
    mma = AdvancedHMC.MassMatrixAdaptor(metric)
    ssa = AdvancedHMC.StepSizeAdaptor(target_acceptance, integrator)
    return AdvancedHMC.NaiveHMCAdaptor(mma, ssa)
end

function ahmc_adaptor(
    tuning::StanLikeTuning,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator,
    θ_init::AbstractVector{<:Real},
    target_acceptance::Real
)
    T = eltype(θ_init)
    mma = AdvancedHMC.MassMatrixAdaptor(metric)
    ssa = AdvancedHMC.StepSizeAdaptor(T(target_acceptance), integrator)
    stan_adaptor = AdvancedHMC.StanHMCAdaptor(
        mma, ssa,
        init_buffer = Int(tuning.initial_bufsize), term_buffer = Int(tuning.term_bufsize), window_size = Int(tuning.window_size)
    )
    return stan_adaptor
end
