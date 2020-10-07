abstract type HMCAdaptor end

struct NoAdaptor <: HMCAdaptor end

struct MassMatrixAdaptor <: HMCAdaptor end

@with_kw struct StepSizeAdaptor <: HMCAdaptor
    target_acceptance::Float64 = 0.8
end

@with_kw struct NaiveHMCAdaptor <: HMCAdaptor
    target_acceptance::Float64 = 0.8
end

@with_kw struct StanHMCAdaptor <: HMCAdaptor
    target_acceptance::Float64 = 0.8
    n_adapts::Int64 = 500
end


function AHMCAdaptor(
    adaptor::NoAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.NoAdaptation()
end


function AHMCAdaptor(
    adaptor::MassMatrixAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.MassMatrixAdaptor(metric)
end


function AHMCAdaptor(
    adaptor::StepSizeAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
)
    return AdvancedHMC.StepSizeAdaptor(adaptor.target_acceptance, integrator)
end


function AHMCAdaptor(
    adaptor::NaiveHMCAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator
)
    mma = AdvancedHMC.MassMatrixAdaptor(metric)
    ssa = AdvancedHMC.StepSizeAdaptor(adaptor.target_acceptance, integrator)
    return AdvancedHMC.NaiveHMCAdaptor(mma, ssa)
end


function AHMCAdaptor(
    adaptor::StanHMCAdaptor,
    metric::AdvancedHMC.AbstractMetric,
    integrator::AdvancedHMC.AbstractIntegrator,
)
    mma = AdvancedHMC.MassMatrixAdaptor(metric)
    ssa = AdvancedHMC.StepSizeAdaptor(adaptor.target_acceptance, integrator)
    stan_adaptor = AdvancedHMC.StanHMCAdaptor(mma, ssa)

    # Initialize state using Stan defaults
    # See: https://mc-stan.org/docs/2_18/reference-manual/hmc-algorithm-parameters.html
    AdvancedHMC.initialize!(stan_adaptor.state, 75, 50, 25, Int(adaptor.n_adapts))
    return stan_adaptor
end
