# This file is a part of BAT.jl, licensed under the MIT License (MIT).
@doc doc"""
    TruncatedDensity

Constructor:

    TruncatedDensity(D<:AbstractDensity, B<:VarVolumeBounds)

*BAT-internal, not part of stable public API.*

A density truncated to given bounds, forced to effectively zero outside of
those bounds. In contrast to a truncated distribution, the density is
*not* renormalized.
"""
struct TruncatedDensity{D<:AbstractDensity,B<:AbstractVarBounds,T<:Real} <: AbstractDensity
    density::D
    bounds::B
    logscalecorr::T
end

Base.parent(density::TruncatedDensity) = density.density

var_bounds(density::TruncatedDensity) = density.bounds

ValueShapes.varshape(density::TruncatedDensity) = varshape(parent(density))


function eval_logval(
    density::TruncatedDensity,
    v::Any,
    T::Type{<:Real} = density_logval_type(v);
    use_bounds::Bool = true,
    strict::Bool = false
)
    v_shaped = reshape_variate(varshape(density), v)
    if use_bounds && !variate_is_inbounds(density, v_shaped, strict)
        return log_zero_density(T)
    end

    parent_logval = eval_logval(
        parent(density), v_shaped,
        use_bounds = false, strict = false
    )  
    
    parent_logval + density.logscalecorr
end


function eval_logval_unchecked(density::TruncatedDensity, v::Any)
    eval_logval(
        density, v,
        use_bounds = false, strict = false
    )
end



@doc doc"""
    BAT.truncate_density(density::AbstractDensity, bounds::AbstractVarBounds)::AbstractDensity

*BAT-internal, not part of stable public API.*

Truncate `density` to `bounds`, the resulting density will be effectively
zero outside of those bounds. In contrast `Distributions.truncated`, `truncate_density`
does *not* renormalize the density.

Currently implemented for `BAT.HyperRectBounds` only.
"""
function truncate_density end


function truncate_density(density::AbstractPosteriorDensity, bounds::AbstractVarBounds)
    old_prior = getprior(density)
    old_bounds = var_bounds(old_prior)
    new_bounds = ismissing(old_bounds) ? bounds : var_bounds(old_prior) ∩ bounds
    newprior = truncate_density(getprior(density), new_bounds)
    PosteriorDensity(getlikelihood(density), newprior)
end


function truncate_density(density::ConstDensity, bounds::AbstractVarBounds)
    old_bounds = var_bounds(density)
    new_bounds = ismissing(old_bounds) ? bounds : var_bounds(density) ∩ bounds
    ConstDensity(density.value, new_bounds)
end


function truncate_density(density::DistributionDensity, bounds::HyperRectBounds)
    old_bounds = var_bounds(density)
    new_bounds = ismissing(old_bounds) ? bounds : var_bounds(density) ∩ bounds
    interval_bounds = Interval.(new_bounds.vol.lo, new_bounds.vol.hi)
    dist = density.dist
    r = truncate_dist_hard(dist, interval_bounds)
    TruncatedDensity(DistributionDensity(r.dist), new_bounds, r.logscalecorr)
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



@doc doc"""
    BAT.truncate_dist_hard(dist::Distribution{Univariate}, bounds::Interval)::Distribution{Univariate}
    BAT.truncate_dist_hard(dist::Distribution{Multivariate}, bounds::AbstractArray{<:Interval})::Distribution{Multivariate}

*BAT-internal, not part of stable public API.*

Generalized variant of `Distributions.truncated` - also handles multivariate
distributions and operates on a best-effort basis: If distributions cannot
be truncated, may return the original distribution.

Returns a `NamedTuple`

```julia
    (dist = trunc_dist, logscalecorr = logscalecorr)
```

with the truncated distribution and the log-PDF amplitude difference to
the original (see [`BAT.trunc_logpdf_ratio`](@ref)).

Mainly used to implement [`BAT.truncate_density`](@ref).
"""
function truncate_dist_hard end

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
    logscalecorr = trunc_logpdf_ratio(dist, trunc_dist)
    return (dist = trunc_dist, logscalecorr = logscalecorr)
end


function truncate_dist_hard(dist::Distributions.Truncated, bounds::Interval)
    # Note: Assumes that dist is result of trunctate, not of truncate_dist_hard:
    untrunc_dist = dist.untruncated

    min_lo = minimum(untrunc_dist)
    max_hi = maximum(untrunc_dist)
    lo = clamp(max(minimum(bounds), dist.lower), min_lo, max_hi)
    hi = clamp(max(lo, min(maximum(bounds), dist.upper)), min_lo, max_hi)
    trunc_dist = truncated(untrunc_dist, lo, hi)
    logscalecorr = trunc_logpdf_ratio(dist, trunc_dist)
    return (dist = trunc_dist, logscalecorr = logscalecorr)
end


function truncate_dist_hard(d::Product, bounds::AbstractArray{<:Interval})
    @argcheck length(eachindex(bounds)) == length(d)
    r = truncate_dist_hard.(d.v, bounds)
    trunc_dists = map(x -> x.dist, r)
    logscalecorr = sum(x.logscalecorr for x in r)

    return (dist = Product(trunc_dists), logscalecorr = logscalecorr)
end


function truncate_dist_hard(dist::ConstValueDist, bounds::AbstractVector{<:Interval})
    @argcheck length(eachindex(bounds)) == 0
    (dist = dist, logscalecorr = 0)
end

function truncate_dist_hard(dist::NamedTupleDist{names}, bounds::AbstractArray{<:Interval}) where names
    @argcheck length(eachindex(bounds)) == totalndof(varshape(dist))
    distributions = values(dist)
    accessors = values(varshape(dist))

    r = map((dist, acc) -> truncate_dist_hard(dist, view(bounds, ValueShapes.view_idxs(eachindex(bounds), acc))), distributions, accessors)
    trunc_dist = NamedTupleDist(NamedTuple{names}(map(x -> x.dist, r)))
    logscalecorr = sum(map(x -> x.logscalecorr, r))
    (dist = trunc_dist, logscalecorr = logscalecorr)
end


@doc doc"""
    BAT.trunc_logpdf_ratio(orig_dist::Distribution{TP}, trunc_dist::Distribution{TP})::AbstractFloat

*BAT-internal, not part of stable public API.*

Computes the log-ratio between the amplitude of the PDF of a truncated
distribution and the original (untruncted) distribution, within the support
of the truncated one.

The PDF of both distributions must have the same shape within the support of
`trunc_dist` and may only differ in amplitude.

Mainly used to implement [`BAT.truncate_density`](@ref), in conjunction with
[`BAT.truncate_dist_hard`](@ref). The result contributes to the `logscalecorr`
factor of a [`TruncatedDensity`] that uses truncated distributions internally,
to ensure the density does not get renormalized.
"""
function trunc_logpdf_ratio end


function trunc_logpdf_ratio(orig_dist::Distribution, trunc_dist::Distribution)
    _trunc_logpdf_ratio_fallback(orig_dist, trunc_dist)
end

function _trunc_logpdf_ratio_fallback(orig_dist::Distribution, trunc_dist::Distribution)
    x = mean(trunc_dist)
    +logpdf(orig_dist, x) - logpdf(trunc_dist, x)
end


function trunc_logpdf_ratio(orig_dist::Distributions.Truncated, trunc_dist::Distributions.Truncated)
    T = promote_type(typeof(trunc_dist.logtp), typeof(orig_dist.logtp))
    if orig_dist.untruncated == trunc_dist.untruncated
        convert(T, trunc_dist.logtp - orig_dist.logtp)
    else
        convert(T, _trunc_logpdf_ratio_fallback(orig_dist, trunc_dist))
    end
end


function trunc_logpdf_ratio(orig_dist::Distribution, trunc_dist::Distributions.Truncated)
    T = typeof(trunc_dist.logtp)
    if orig_dist == trunc_dist.untruncated
        trunc_dist.logtp
    else
        convert(T, _trunc_logpdf_ratio_fallback(orig_dist, trunc_dist))
    end
end
