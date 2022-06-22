# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DistMeasure{
    D<:ContinuousDistribution,
    B<:AbstractVarBounds
} <: DistLikeMeasure
    dist::D
    bounds::B
end

DistMeasure(d::Distribution) = DistMeasure(d, dist_param_bounds(d))
DistMeasure(d::DistributionMeasure) = DistMeasure(d.d, dist_param_bounds(d.d))

Base.convert(::Type{AbstractMeasureOrDensity}, d::ContinuousDistribution) = DistMeasure(d)
Base.convert(::Type{DistLikeMeasure}, d::ContinuousDistribution) = DistMeasure(d)

Base.convert(::Type{Distribution}, d::DistMeasure) = d.dist
Base.convert(::Type{ContinuousDistribution}, d::DistMeasure) = d.dist

Base.convert(::Type{AbstractMeasureOrDensity}, d::DistributionMeasure) = DistMeasure(d)
Base.convert(::Type{DistLikeMeasure}, d::DistributionMeasure) = DistMeasure(d)


Base.parent(density::DistMeasure) = density.dist


Base.:(==)(a::DistMeasure, b::DistMeasure) = a.dist == b.dist && a.bounds == b.bounds


function DensityInterface.logdensityof(density::DistMeasure{<:Distribution{Univariate,Continuous}}, v::Real)
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

DensityInterface.logdensityof(density::DistMeasure, v::Any) = Distributions.logpdf(density.dist, v)


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


var_bounds(density::DistMeasure) = density.bounds


dist_param_bounds(d::Distribution{Univariate,Continuous}) =
    HyperRectBounds([minimum(d)], [maximum(d)])

dist_param_bounds(d::Distribution{Multivariate,Continuous}) =
    HyperRectBounds(fill(_default_PT(-Inf), length(d)), fill(_default_PT(+Inf), length(d)))

dist_param_bounds(d::StandardUniformDist) =
    HyperRectBounds(fill(_default_PT(Float32(0)), length(d)), fill(_default_PT(Float32(1)), length(d)))

dist_param_bounds(d::StandardNormalDist) =
    HyperRectBounds(fill(_default_PT(Float32(-Inf)), length(d)), fill(_default_PT(Float32(+Inf)), length(d)))

dist_param_bounds(d::ReshapedDist) = dist_param_bounds(unshaped(d))

dist_param_bounds(d::Product{Continuous}) =
    HyperRectBounds(minimum.(d.v), maximum.(d.v))

dist_param_bounds(d::ConstValueDist) = HyperRectBounds(Int32[], Int32[])

dist_param_bounds(d::NamedTupleDist) = vcat(map(x -> dist_param_bounds(x), values(d))...)
dist_param_bounds(d::ValueShapes.UnshapedNTD) = dist_param_bounds(d.shaped)

dist_param_bounds(d::HierarchicalDistribution) =
    HyperRectBounds(fill(_default_PT(-Inf), length(d)), fill(_default_PT(+Inf), length(d)))
