# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DistributionDensity{
    D<:Distribution{Multivariate,Continuous},
    B<:AbstractVarBounds
} <: DistLikeDensity
    dist::D
    bounds::B
end

DistributionDensity(d::Distribution) = DistributionDensity(d, dist_param_bounds(d))

DistributionDensity(h::Histogram) = DistributionDensity(EmpiricalDistributions.MvBinnedDist(h))


Base.convert(::Type{AbstractDensity}, d::Distribution{Multivariate,Continuous}) =
    DistributionDensity(d)
    
Base.convert(::Type{DistLikeDensity}, d::Distribution{Multivariate,Continuous}) =
    DistributionDensity(d)

Base.convert(::Type{AbstractDensity}, h::Histogram) = DistributionDensity(h)
Base.convert(::Type{DistLikeDensity}, h::Histogram) = DistributionDensity(h)


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


dist_param_bounds(d::Distribution{Univariate,Continuous}) = HyperRectBounds([minimum(d)], [maximum(d)], reflective_bounds)
dist_param_bounds(d::Distribution{Multivariate,Continuous}) = HyperRectBounds(fill(_default_PT(-Inf), length(d)), fill(_default_PT(+Inf), length(d)), hard_bounds)
dist_param_bounds(d::Product{Continuous}) = HyperRectBounds(minimum.(d.v), maximum.(d.v), reflective_bounds)

dist_param_bounds(d::ConstValueDist) = HyperRectBounds(Int32[], Int32[], hard_bounds)
dist_param_bounds(d::NamedTupleDist) = vcat(map(dist_param_bounds, values(d))...)

function dist_param_bounds(d::EmpiricalDistributions.MvBinnedDist{T, N}) where {T, N}
    left_bounds  = T[map(first, d.h.edges)...]
    right_bounds = T[map(e -> prevfloat(last(e)), d.h.edges)...]
    bt = fill(reflective_bounds, length(left_bounds))
    HyperRectBounds{T}(HyperRectVolume{T}(left_bounds, right_bounds), bt)
end
