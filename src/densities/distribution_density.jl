# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DistributionDensity{
    D<:Distribution{Multivariate,Continuous},
    B<:AbstractVarBounds
} <: DistLikeDensity
    dist::D
    bounds::B
end

DistributionDensity(d::Distribution) = DistributionDensity(d, dist_param_bounds(d))


Base.convert(::Type{AbstractDensity}, d::Distribution{Multivariate,Continuous}) =
    DistributionDensity(d)
    
Base.convert(::Type{DistLikeDensity}, d::Distribution{Multivariate,Continuous}) =
    DistributionDensity(d)


Base.parent(density::DistributionDensity) = density.dist


function density_logval(
    density::DistributionDensity,
    v::Any
)
    Distributions.logpdf(density.dist, v)
end

ValueShapes.varshape(density::DistributionDensity) = varshape(density.dist)

Distributions.sampler(density::DistributionDensity) = bat_sampler(parent(density))

Random.Sampler(rng::AbstractRNG, density::DistributionDensity, repetition::Val{1}) = sampler(density)

Statistics.cov(density::DistributionDensity) = cov(density.dist)


var_bounds(density::DistributionDensity) = density.bounds


dist_param_bounds(d::Distribution{Univariate,Continuous}) = HyperRectBounds([quantile(d, 0f0)], [quantile(d, 1f0)], reflective_bounds)
dist_param_bounds(d::Distribution{Multivariate,Continuous}) = HyperRectBounds(fill(_default_PT(-Inf), length(d)), fill(_default_PT(+Inf), length(d)), hard_bounds)
dist_param_bounds(d::Product{Continuous}) = HyperRectBounds(quantile.(d.v, 0f0), quantile.(d.v, 1f0), reflective_bounds)

dist_param_bounds(d::ConstValueDist) = HyperRectBounds(Int32[], Int32[], hard_bounds)
dist_param_bounds(d::NamedTupleDist) = vcat(map(dist_param_bounds, values(d))...)
