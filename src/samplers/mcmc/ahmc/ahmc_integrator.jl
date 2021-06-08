abstract type HMCIntegrator end

@with_kw mutable struct LeapfrogIntegrator{T<:Float64} <: HMCIntegrator
    step_size::T = 0.0
end

@with_kw mutable struct JitteredLeapfrogIntegrator{T<:Float64} <: HMCIntegrator
    step_size::T = 0.0
    jitter_rate::T = 1.0
end

@with_kw mutable struct TemperedLeapfrogIntegrator{T<:Float64} <: HMCIntegrator
    step_size::T = 0.0
    tempering_rate::T = 1.05
end



function AHMCIntegrator(integrator::LeapfrogIntegrator)
    return AdvancedHMC.Leapfrog(integrator.step_size)
end

function AHMCIntegrator(integrator::JitteredLeapfrogIntegrator)
    return AdvancedHMC.JitteredLeapfrog(integrator.step_size, integrator.jitter_rate)
end

function AHMCIntegrator(integrator::TemperedLeapfrogIntegrator)
    return AdvancedHMC.TemperedLeapfrog(integrator.step_size, integrator.tempering_rate)
end
