abstract type HMCIntegrator end

function _ahmc_step_size(rng::AbstractRNG, integrator::HMCIntegrator, hamiltonian::AdvancedHMC.Hamiltonian, x_init::AbstractVector{<:Real})
    if isnan(integrator.step_size)
        oftype(integrator.step_size, AdvancedHMC.find_good_stepsize(rng, hamiltonian, x_init, max_n_iters = 100))
    else
        integrator.step_size
    end
end



@with_kw struct LeapfrogIntegrator <: HMCIntegrator
    step_size::Float64 = NaN
end

function ahmc_integrator(rng::AbstractRNG, integrator::LeapfrogIntegrator, hamiltonian::AdvancedHMC.Hamiltonian, x_init::AbstractVector{<:Real})
    return AdvancedHMC.Leapfrog(_ahmc_step_size(rng, integrator, hamiltonian, x_init))
end



@with_kw struct JitteredLeapfrogIntegrator <: HMCIntegrator
    step_size::Float64 = NaN
    jitter_rate::Float64 = 1.0
end

function ahmc_integrator(rng::AbstractRNG, integrator::JitteredLeapfrogIntegrator, hamiltonian::AdvancedHMC.Hamiltonian, x_init::AbstractVector{<:Real})
    return AdvancedHMC.JitteredLeapfrog(_ahmc_step_size(rng, integrator, hamiltonian, x_init), integrator.jitter_rate)
end



@with_kw struct TemperedLeapfrogIntegrator <: HMCIntegrator
    step_size::Float64 = NaN
    tempering_rate::Float64 = 1.05
end

function ahmc_integrator(rng::AbstractRNG, integrator::TemperedLeapfrogIntegrator, hamiltonian::AdvancedHMC.Hamiltonian, x_init::AbstractVector{<:Real})
    return AdvancedHMC.TemperedLeapfrog(_ahmc_step_size(rng, integrator, hamiltonian, x_init), integrator.tempering_rate)
end
