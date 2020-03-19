export Preconditioner
export NesterovDualAveraging
export NaiveHMCAdaptor
export StanHMCAdaptor


abstract type AHMCAdaptor end

struct Preconditioner <: AHMCAdaptor end

struct NesterovDualAveraging <: AHMCAdaptor
    δ::Real
    NesterovDualAveraging(; δ=0.8) = new(δ)
end

struct NaiveHMCAdaptor <: AHMCAdaptor
    δ::Real
    NaiveHMCAdaptor(; δ=0.8) = new(δ)
end

struct StanHMCAdaptor <: AHMCAdaptor
    δ::Real
    StanHMCAdaptor(; δ=0.8) = new(δ)
end



function get_AHMCAdaptor(
    adaptor::Preconditioner,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
    )
    return AdvancedHMC.Preconditioner(metric)
end

function get_AHMCAdaptor(
    adaptor::NesterovDualAveraging,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
    )
    return AdvancedHMC.NesterovDualAveraging(adaptor.δ, integrator)
end

function get_AHMCAdaptor(
    adaptor::NaiveHMCAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
    )
    pc = AdvancedHMC.Preconditioner(metric)
    da = AdvancedHMC.NesterovDualAveraging(adaptor.δ, integrator)
    return AdvancedHMC.NaiveHMCAdaptor(pc, da)
end

function get_AHMCAdaptor(
    adaptor::StanHMCAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
    )
    pc = AdvancedHMC.Preconditioner(metric)
    da = AdvancedHMC.NesterovDualAveraging(adaptor.δ, integrator)
    return AdvancedHMC.StanHMCAdaptor(pc, da)
end
