# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DistributionDensity{
    D<:Distribution{Multivariate,Continuous},
    B<:AbstractVarBounds
} <: DistLikeDensity
    dist::D
    bounds::B
end

DistributionDensity(d::Distribution; bounds_type::BoundsType = hard_bounds) =
    DistributionDensity(d, dist_param_bounds(d, bounds_type))

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


dist_param_bounds(d::Distribution{Univariate,Continuous}, bounds_type::BoundsType) =
    HyperRectBounds([minimum(d)], [maximum(d)], bounds_type)

dist_param_bounds(d::Distribution{Multivariate,Continuous}, bounds_type::BoundsType) =
    HyperRectBounds(fill(_default_PT(-Inf), length(d)), fill(_default_PT(+Inf), length(d)), bounds_type)

dist_param_bounds(d::Product{Continuous}, bounds_type::BoundsType) =
    HyperRectBounds(minimum.(d.v), maximum.(d.v), bounds_type)

dist_param_bounds(d::ConstValueDist, bounds_type::BoundsType) = HyperRectBounds(Int32[], Int32[], bounds_type)
dist_param_bounds(d::NamedTupleDist, bounds_type::BoundsType) = vcat(map(x -> dist_param_bounds(x, bounds_type), values(d))...)

function dist_param_bounds(d::EmpiricalDistributions.MvBinnedDist{T, N}, bounds_type::BoundsType) where {T, N}
    left_bounds  = T[map(first, d.h.edges)...]
    right_bounds = T[map(e -> prevfloat(last(e)), d.h.edges)...]
    bt = fill(bounds_type, length(left_bounds))
    HyperRectBounds{T}(HyperRectVolume{T}(left_bounds, right_bounds), bt)
end


function estimate_finite_bounds(ntd::NamedTupleDist; bounds_type::BoundsType=hard_bounds)
    bounds = vcat([estimate_finite_bounds(d) for d in values(ntd)]...)
    lo = [b[1] for b in bounds]
    hi = [b[2] for b in bounds]
    HyperRectBounds(lo, hi, bounds_type)
end

function estimate_finite_bounds(d::Distribution{Univariate})
    lo, hi = minimum(d), maximum(d)

    if isinf(lo)
        lo = typeof(lo)(quantile(d, 0.00001))
    end

    if isinf(hi)
        hi = typeof(hi)(quantile(d, 0.99999))
    end

    return lo, hi
end

function estimate_finite_bounds(d::Product)
    bounds = estimate_finite_bounds.(d.v)
    return vcat(bounds...)
end
