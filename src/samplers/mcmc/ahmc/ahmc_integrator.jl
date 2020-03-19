export Leapfrog
export JitteredLeapfrog
export TemperedLeapfrog


abstract type AHMCIntegrator end

mutable struct Leapfrog <: AHMCIntegrator
    ϵ::Real # step size
    Leapfrog(; ϵ=0) = new(ϵ)
end

mutable struct JitteredLeapfrog <: AHMCIntegrator
    ϵ::Real
    n::Real # jitter rate
    JitteredLeapfrog(; ϵ=0, n=1.0) = new(ϵ, n)
end

mutable struct TemperedLeapfrog <: AHMCIntegrator
    ϵ::Real
    a::Real # tempering rate
    TemperedLeapfrog(; ϵ=0, a=1.05) = new(ϵ, a)
end



function get_AHMCintegrator(integrator::Leapfrog)
    return AdvancedHMC.Leapfrog(integrator.ϵ)
end

function get_AHMCintegrator(integrator::JitteredLeapfrog)
    return AdvancedHMC.JitteredLeapfrog(integrator.ϵ, integrator.n)
end

function get_AHMCintegrator(integrator::TemperedLeapfrog)
    return AdvancedHMC.TemperedLeapfrog(integrator.ϵ, integrator.a)
end
