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
    @unpack step_size = integrator
    return AdvancedHMC.Leapfrog(step_size)
end

function AHMCIntegrator(integrator::JitteredLeapfrogIntegrator)
    @unpack step_size, jitter_rate = integrator
    return AdvancedHMC.JitteredLeapfrog(step_size, jitter_rate)
end

function AHMCIntegrator(integrator::TemperedLeapfrogIntegrator)
    @unpack step_size, tempering_rate = integrator
    return AdvancedHMC.TemperedLeapfrog(step_size, tempering_rate)
end
