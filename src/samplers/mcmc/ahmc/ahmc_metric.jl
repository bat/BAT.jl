abstract type HMCMetric end
struct DiagEuclideanMetric <: HMCMetric end
struct UnitEuclideanMetric <: HMCMetric end
struct DenseEuclideanMetric <: HMCMetric end



function AHMCMetric(metric::DiagEuclideanMetric, dim::Integer)
    return AdvancedHMC.DiagEuclideanMetric(dim)
end

function AHMCMetric(metric::UnitEuclideanMetric, dim::Integer)
    return AdvancedHMC.UnitEuclideanMetric(dim)
end

function AHMCMetric(metric::DenseEuclideanMetric, dim::Integer)
    return AdvancedHMC.DenseEuclideanMetric(dim)
end
