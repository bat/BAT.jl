# This file is a part of BAT.jl, licensed under the MIT License (MIT).
@doc doc"""
    TruncatedDensity

Constructor:

    TruncatedDensity(D<:AbstractDensity, B<:VarVolumeBounds)

*BAT-internal, not part of stable public API.*

Density with specified bounds.
"""
struct TruncatedDensity{D<:AbstractDensity,B<:AbstractVarBounds,T<:Real} <: AbstractDensity
    density::D
    bounds::B
    logrenorm::T
end

Base.parent(density::TruncatedDensity) = density.density

var_bounds(density::TruncatedDensity) = density.bounds

ValueShapes.varshape(density::TruncatedDensity) = varshape(parent(density))

function density_logval(density::TruncatedDensity, v::Any)
    # ToDo: Enforce bounds - currently have to trust BAT.eval_density_logval! to do it,
    # since value shape is not available here.
    density_logval(parent(density), v) + density.logrenorm
end



function truncate_density(density::AbstractPosteriorDensity, bounds::AbstractVarBounds)
    old_prior = getprior(density)
    old_bounds = var_bounds(old_prior)
    new_bounds = ismissing(old_bounds) ? bounds : var_bounds(old_prior) ∩ bounds
    newprior = truncate_density(getprior(density), new_bounds)
    PosteriorDensity(getlikelihood(density), newprior)
end


function truncate_density(density::DistributionDensity, bounds::VarVolumeBounds)
    old_bounds = var_bounds(density)
    new_bounds = ismissing(old_bounds) ? bounds : var_bounds(density) ∩ bounds
    interval_bounds = Interval.(new_bounds.vol.lo, new_bounds.vol.hi)
    dist = density.dist
    r = truncate_dist_hard(dist, interval_bounds)
    TruncatedDensity(DistributionDensity(r.dist), new_bounds, r.logrenorm)
end

function truncate_density(density::TruncatedDensity{<:DistributionDensity}, bounds::AbstractVarBounds)
    # Tricky: How to decide whether to propagate renormalization of inner densities/distributions ?
    @assert false # Not implemented yet
end

# ToDo: Reject out-of-bounds samples, in case density contains distributions that can't be truncated (e.g. multivariate)
Distributions.sampler(density::TruncatedDensity{<:DistributionDensity}) = Distributions.sampler(parent(density))

function Statistics.cov(density::TruncatedDensity{<:DistributionDensity})
    cov(nestedview(rand(bat_determ_rng(), sampler(density), 10^5)))
end



function truncate_dist_hard(dist::Distribution{Univariate}, bounds::AbstractArray{<:Interval})
    @argcheck length(eachindex(bounds)) == 1
    truncate_dist_hard(dist, first(bounds))
end


function truncate_dist_hard(dist::Distribution{Univariate}, bounds::Interval)
    min_lo = minimum(dist)
    max_hi = maximum(dist)
    lo = clamp(minimum(bounds), min_lo, max_hi)
    hi = clamp(max(lo, maximum(bounds)), min_lo, max_hi)

    trunc_dist = truncated(dist, lo, hi)
    logrenorm = trunc_dist.logtp
    return (dist = trunc_dist, logrenorm = logrenorm)
end


function truncate_dist_hard(dist::Truncated, bounds::Interval)
    # Note: Assumes that dist is result of trunctate, not of truncate_dist_hard:
    untrunc_dist = dist.untruncated

    min_lo = minimum(untrunc_dist)
    max_hi = maximum(untrunc_dist)
    lo = clamp(max(minimum(bounds), dist.lower), min_lo, max_hi)
    hi = clamp(max(lo, min(maximum(bounds), dist.upper)), min_lo, max_hi)
    trunc_dist = truncated(untrunc_dist, lo, hi)
    logrenorm = trunc_dist.logtp - dist.logtp
    return (dist = trunc_dist, logrenorm = logrenorm)
end


function truncate_dist_hard(d::Product, bounds::AbstractArray{<:Interval})
    @argcheck length(eachindex(bounds)) == length(d)
    r = truncate_dist_hard.(d.v, bounds)
    trunc_dists = map(x -> x.dist, r)
    logrenorm = sum(x.logrenorm for x in r)

    return (dist = Product(trunc_dists), logrenorm = logrenorm)
end


function truncate_dist_hard(dist::ConstValueDist, bounds::AbstractVector{<:Interval})
    @argcheck length(eachindex(bounds)) == 0
    (dist = dist, logrenorm = 0)
end

function truncate_dist_hard(dist::NamedTupleDist{names}, bounds::AbstractArray{<:Interval}) where names
    @argcheck length(eachindex(bounds)) == length(dist)
    distributions = values(dist)
    accessors = values(varshape(dist))

    r = map((dist, acc) -> truncate_dist_hard(dist, view(bounds, ValueShapes.view_idxs(eachindex(bounds), acc))), distributions, accessors)
    trunc_dist = NamedTupleDist(NamedTuple{names}(map(x -> x.dist, r)))
    logrenorm = sum(map(x -> x.logrenorm, r))
    (dist = trunc_dist, logrenorm = logrenorm)
end
