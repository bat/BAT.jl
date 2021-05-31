# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DistributionDensity{
    D<:ContinuousDistribution,
    B<:AbstractVarBounds
} <: DistLikeDensity
    dist::D
    bounds::B
end

DistributionDensity(d::Distribution) = DistributionDensity(d, dist_param_bounds(d))

Base.convert(::Type{AbstractDensity}, d::ContinuousDistribution) = DistributionDensity(d)
Base.convert(::Type{DistLikeDensity}, d::ContinuousDistribution) = DistributionDensity(d)

Base.convert(::Type{Distribution}, d::DistributionDensity) = d.dist
Base.convert(::Type{ContinuousDistribution}, d::DistributionDensity) = d.dist


Base.parent(density::DistributionDensity) = density.dist


function eval_logval_unchecked(density::DistributionDensity{<:Distribution{Univariate,Continuous}}, v::Real)
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

eval_logval_unchecked(density::DistributionDensity, v::Any) = Distributions.logpdf(density.dist, v)


ValueShapes.varshape(density::DistributionDensity) = varshape(density.dist)

ValueShapes.unshaped(density::DistributionDensity) = DistributionDensity(unshaped(density.dist))

(shape::AbstractValueShape)(density::DistributionDensity) = DistributionDensity(shape(density.dist))

# For user convenience, don't use within BAT:
@inline Random.rand(rng::AbstractRNG, density::DistributionDensity) = rand(rng, density.dist)
@inline Random.rand(rng::AbstractRNG, density::DistributionDensity, dims::Dims) = rand(rng, density.dist, dims)
@inline Random.rand(rng::AbstractRNG, density::DistributionDensity, dims::Integer...) = rand(rng, density.dist, dims...)

Distributions.sampler(density::DistributionDensity) = Distributions.sampler(density.dist)
bat_sampler(density::DistributionDensity) = bat_sampler(density.dist)

Statistics.cov(density::DistributionDensity{<:MultivariateDistribution}) = cov(density.dist)


var_bounds(density::DistributionDensity) = density.bounds


dist_param_bounds(d::Distribution{Univariate,Continuous}) =
    HyperRectBounds([minimum(d)], [maximum(d)])

dist_param_bounds(d::Distribution{Multivariate,Continuous}) =
    HyperRectBounds(fill(_default_PT(-Inf), length(d)), fill(_default_PT(+Inf), length(d)))

dist_param_bounds(d::ReshapedDist) = dist_param_bounds(unshaped(d))

dist_param_bounds(d::StandardMvUniform) =
    HyperRectBounds(fill(_default_PT(Float32(0)), length(d)), fill(_default_PT(Float32(1)), length(d)))

dist_param_bounds(d::Product{Continuous}) =
    HyperRectBounds(minimum.(d.v), maximum.(d.v))

dist_param_bounds(d::ConstValueDist) = HyperRectBounds(Int32[], Int32[])

dist_param_bounds(d::NamedTupleDist) = vcat(map(x -> dist_param_bounds(x), values(d))...)
dist_param_bounds(d::ValueShapes.UnshapedNTD) = dist_param_bounds(d.shaped)

dist_param_bounds(d::HierarchicalDistribution) =
    HyperRectBounds(fill(_default_PT(-Inf), length(d)), fill(_default_PT(+Inf), length(d)))



const StandardUniformDensity = Union{
    DistributionDensity{<:StandardUvUniform},
    DistributionDensity{<:StandardMvUniform}
}

const StandardNormalDensity= Union{
    DistributionDensity{<:StandardUvNormal},
    DistributionDensity{<:StandardMvNormal}
}
