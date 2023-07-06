# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Integrator ==============================================

function _ahmc_step_size(rng::AbstractRNG, integrator::HMCIntegrator, hamiltonian::AdvancedHMC.Hamiltonian, x_init::AbstractVector{<:Real})
    if isnan(integrator.step_size)
        oftype(integrator.step_size, AdvancedHMC.find_good_stepsize(rng, hamiltonian, x_init, max_n_iters = 100))
    else
        integrator.step_size
    end
end

function ahmc_integrator(rng::AbstractRNG, integrator::LeapfrogIntegrator, hamiltonian::AdvancedHMC.Hamiltonian, x_init::AbstractVector{<:Real})
    return AdvancedHMC.Leapfrog(_ahmc_step_size(rng, integrator, hamiltonian, x_init))
end

function ahmc_integrator(rng::AbstractRNG, integrator::JitteredLeapfrogIntegrator, hamiltonian::AdvancedHMC.Hamiltonian, x_init::AbstractVector{<:Real})
    return AdvancedHMC.JitteredLeapfrog(_ahmc_step_size(rng, integrator, hamiltonian, x_init), integrator.jitter_rate)
end

function ahmc_integrator(rng::AbstractRNG, integrator::TemperedLeapfrogIntegrator, hamiltonian::AdvancedHMC.Hamiltonian, x_init::AbstractVector{<:Real})
    return AdvancedHMC.TemperedLeapfrog(_ahmc_step_size(rng, integrator, hamiltonian, x_init), integrator.tempering_rate)
end


# Proposal ==============================================

function ahmc_proposal(
    proposal::FixedStepNumber,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.StaticTrajectory(integrator, proposal.nsteps)
end

function ahmc_proposal(
    proposal::FixedTrajectoryLength,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.HMCDA(integrator, proposal.trajectory_length)
end

function ahmc_proposal(
    proposal::NUTSProposal,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.NUTS{AdvancedHMC.MultinomialTS, AdvancedHMC.ClassicNoUTurn}(integrator)
end


# Metric ==============================================

function ahmc_metric(metric::DiagEuclideanMetric, dim::Integer)
    return AdvancedHMC.DiagEuclideanMetric(dim)
end

function ahmc_metric(metric::UnitEuclideanMetric, dim::Integer)
    return AdvancedHMC.UnitEuclideanMetric(dim)
end

function ahmc_metric(metric::DenseEuclideanMetric, dim::Integer)
    return AdvancedHMC.DenseEuclideanMetric(dim)
end



# Tuning ==============================================


function ahmc_adaptor(
    tuning::MassMatrixAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.MassMatrixAdaptor(metric)
end

function ahmc_adaptor(
    tuning::StepSizeAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.StepSizeAdaptor(tuning.target_acceptance, integrator)
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
