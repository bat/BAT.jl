# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DistributionDensity{
    D<:ContinuousDistribution,
    B<:AbstractVarBounds
} <: DistLikeDensity
    dist::D
    bounds::B
end

DistributionDensity(d::Distribution; bounds_type::BoundsType = hard_bounds) =
    DistributionDensity(d, dist_param_bounds(d, bounds_type))

Base.convert(::Type{AbstractDensity}, d::ContinuousDistribution) = DistributionDensity(d)
Base.convert(::Type{DistLikeDensity}, d::ContinuousDistribution) = DistributionDensity(d)

DistributionDensity(h::Histogram) = DistributionDensity(EmpiricalDistributions.MvBinnedDist(h))

Base.convert(::Type{AbstractDensity}, h::Histogram) = DistributionDensity(h)
Base.convert(::Type{DistLikeDensity}, h::Histogram) = DistributionDensity(h)


Base.parent(density::DistributionDensity) = density.dist


eval_logval_unchecked(density::DistributionDensity, v::Any) = Distributions.logpdf(density.dist, v)

eval_logval_unchecked(density::DistributionDensity, v::AbstractVector{<:Real}) = Distributions.logpdf(unshaped(density.dist), v)


ValueShapes.varshape(density::DistributionDensity) = varshape(density.dist)

Distributions.sampler(density::DistributionDensity) = bat_sampler(unshaped(density.dist))


# Random.Sampler(rng::AbstractRNG, density::DistributionDensity, repetition::Val{1}) = sampler(density)

Statistics.cov(density::DistributionDensity) = cov(unshaped(density.dist))


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
    hist = convert(Histogram, d)
    left_bounds  = T[map(first, hist.edges)...]
    right_bounds = T[map(e -> prevfloat(last(e)), hist.edges)...]
    bt = fill(bounds_type, length(left_bounds))
    HyperRectBounds{T}(HyperRectVolume{T}(left_bounds, right_bounds), bt)
end



function estimate_finite_bounds(density::DistributionDensity; bounds_type::BoundsType = hard_bounds)
    return estimate_finite_bounds(density.dist, bounds_type = bounds_type)
end


function estimate_finite_bounds(ntd::NamedTupleDist; bounds_type::BoundsType = hard_bounds)
    bounds = vcat([estimate_finite_bounds(d) for d in values(ntd)]...)
    lo = [b[1] for b in bounds]
    hi = [b[2] for b in bounds]
    HyperRectBounds(lo, hi, bounds_type)
end

function estimate_finite_bounds(d::Distribution{Univariate,Continuous})
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
