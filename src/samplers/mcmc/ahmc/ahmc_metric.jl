export DiagEuclideanMetric
export UnitEuclideanMetric
export DenseEuclideanMetric

abstract type AHMCMetric end
struct DiagEuclideanMetric <: AHMCMetric end
struct UnitEuclideanMetric <: AHMCMetric end
struct DenseEuclideanMetric <: AHMCMetric end



function get_AHMCmetric(metric::DiagEuclideanMetric, dim::Int64)
    return AdvancedHMC.DiagEuclideanMetric(dim)
end

function get_AHMCmetric(metric::UnitEuclideanMetric, dim::Int64)
    return AdvancedHMC.UnitEuclideanMetric(dim)
end

function get_AHMCmetric(metric::DenseEuclideanMetric, dim::Int64)
    return AdvancedHMC.DenseEuclideanMetric(dim)
end
