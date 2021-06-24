abstract type HMCMetric end



struct DiagEuclideanMetric <: HMCMetric end

function ahmc_metric(metric::DiagEuclideanMetric, dim::Integer)
    return AdvancedHMC.DiagEuclideanMetric(dim)
end



struct UnitEuclideanMetric <: HMCMetric end

function ahmc_metric(metric::UnitEuclideanMetric, dim::Integer)
    return AdvancedHMC.UnitEuclideanMetric(dim)
end



struct DenseEuclideanMetric <: HMCMetric end

function ahmc_metric(metric::DenseEuclideanMetric, dim::Integer)
    return AdvancedHMC.DenseEuclideanMetric(dim)
end




