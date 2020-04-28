export DiagEuclideanMetric
export UnitEuclideanMetric
export DenseEuclideanMetric

abstract type HMCMetric end
struct DiagEuclideanMetric <: HMCMetric end
struct UnitEuclideanMetric <: HMCMetric end
struct DenseEuclideanMetric <: HMCMetric end



function AHMCMetric(metric::DiagEuclideanMetric, dim::Int64)
    return AdvancedHMC.DiagEuclideanMetric(dim)
end

function AHMCMetric(metric::UnitEuclideanMetric, dim::Int64)
    return AdvancedHMC.UnitEuclideanMetric(dim)
end

function AHMCMetric(metric::DenseEuclideanMetric, dim::Int64)
    return AdvancedHMC.DenseEuclideanMetric(dim)
end
