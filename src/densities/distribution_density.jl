# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DistMeasure{
    D<:ContinuousDistribution,
    B<:AbstractVarBounds
} <: DistLikeMeasure
    dist::D
    bounds::B
end

MeasureBase.getdof(m::DistMeasure) = eff_totalndof(m.dist)

DistMeasure(d::Distribution) = DistMeasure(d, dist_param_bounds(d))

Base.convert(::Type{BATMeasure}, d::ContinuousDistribution) = DistMeasure(d)
Base.convert(::Type{DistLikeMeasure}, d::ContinuousDistribution) = DistMeasure(d)

Base.convert(::Type{Distribution}, d::DistMeasure) = d.dist
Base.convert(::Type{ContinuousDistribution}, d::DistMeasure) = d.dist


Base.parent(density::DistMeasure) = density.dist


Base.:(==)(a::DistMeasure, b::DistMeasure) = a.dist == b.dist && a.bounds == b.bounds


function logdensityof_batmeasure(density::DistMeasure{<:Distribution{Univariate,Continuous}}, v::Real)
    d = density.dist
    logd = logpdf(d, v)
    R = typeof(logd)
    # ToDo: Move these workarounds somewhere else? Still necessary at all?
    if isnan(logd)
        if isinf(v)
            # Weibull yields NaN logpdf at infinity (Distributions.jl issue #1197), possibly others too,
            # so force to -Inf (there should never be any probability mass at infinity):
            convert(R, -Inf)
        elseif v ≈ minimum(d)
            # Weibull yields NaN logpdf at 0 (Distributions.jl issue #1197), possibly others too,
            # so move an epsilon away from minimum:
            convert(R, logpdf(d, minimum(d) + eps(typeof(v))))
        elseif v ≈ maximum(d)
            # Likewise at maxiumum:
            convert(R, logpdf(d, maximum(d) - eps(typeof(v))))
        else
            logd
        end
    else
        logd
    end
end

logdensityof_batmeasure(density::DistMeasure, v::Any) = Distributions.logpdf(density.dist, v)


ValueShapes.varshape(density::DistMeasure) = varshape(density.dist)

ValueShapes.unshaped(density::DistMeasure) = DistMeasure(unshaped(density.dist))

(shape::AbstractValueShape)(density::DistMeasure) = DistMeasure(shape(density.dist))

# For user convenience, don't use within BAT:
@inline Random.rand(rng::AbstractRNG, density::DistMeasure) = rand(rng, density.dist)
@inline Random.rand(rng::AbstractRNG, density::DistMeasure, dims::Dims) = rand(rng, density.dist, dims)
@inline Random.rand(rng::AbstractRNG, density::DistMeasure, dims::Integer...) = rand(rng, density.dist, dims...)

Distributions.sampler(density::DistMeasure) = Distributions.sampler(density.dist)
bat_sampler(density::DistMeasure) = bat_sampler(density.dist)

Statistics.cov(density::DistMeasure{<:MultivariateDistribution}) = cov(density.dist)



"""
    BAT.AnyIIDSampleable = Union{...}

*BAT-internal, not part of stable public API.*

Union of all distribution/density-like types that BAT can draw i.i.d.
(independent and identically distributed) samples from:

* [`DistLikeMeasure`](@ref)
* `Distributions.Distribution`
"""
const AnyIIDSampleable = Union{
    DistMeasure,
    Distributions.Distribution,
}
