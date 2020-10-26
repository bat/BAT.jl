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


Base.parent(density::DistributionDensity) = density.dist


function eval_logval_unchecked(density::DistributionDensity, v::Any)
    d = density.dist
    logd = logpdf(d, v)
    R = typeof(logd)
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
