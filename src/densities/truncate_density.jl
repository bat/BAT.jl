# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    BAT.truncate_density(density::AbstractDensity, bounds::AbstractArray{<:Interval})::AbstractDensity

*Experimental feature, not part of stable public API.*

Truncate `density` to `bounds`, the resulting density will be effectively
zero outside of those bounds. In contrast `Distributions.truncated`,
`truncate_density` does *not* renormalize the density.

Requires `varshape(density) isa ArrayShape`.

Only supports densities that are essentially products of univariate
distributions, as well as posterior densities with such densities as priors.
"""
function truncate_density end
export truncate_density


function truncate_density(density::AbstractPosteriorDensity, bounds::AbstractArray{<:Interval})
    @argcheck varshape(density) isa ArrayShape
    PosteriorDensity(getlikelihood(density), truncate_density(getprior(density), bounds))
end


function truncate_density(density::DistributionDensity{<:MultivariateDistribution}, bounds::AbstractArray{<:Interval})
    r = truncate_dist_hard(density.dist, bounds)
    RenormalizedDensity(DistributionDensity(r.dist), r.logrenormf)
end

function truncate_density(density::RenormalizedDensity{<:DistributionDensity{<:MultivariateDistribution}}, bounds::AbstractArray{<:Interval})
    r = truncate_dist_hard(density.density.dist, bounds)
    RenormalizedDensity(DistributionDensity(r.dist), r.logrenormf + density.logrenormf)
end



"""
    BAT.truncate_dist_hard(dist::Distribution{Univariate}, bounds::Interval)::Distribution{Univariate}
    BAT.truncate_dist_hard(dist::Distribution{Multivariate}, bounds::AbstractArray{<:Interval})::Distribution{Multivariate}

*BAT-internal, not part of stable public API.*

Generalized variant of `Distributions.truncated` - also handles multivariate
distributions and operates on a best-effort basis: If distributions cannot
be truncated, may return the original distribution.

Returns a `NamedTuple`

```julia
    (dist = trunc_dist, logrenormf = logrenormf)
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
    logrenormf = trunc_logpdf_ratio(dist, trunc_dist)
    return (dist = trunc_dist, logrenormf = logrenormf)
end


function truncate_dist_hard(dist::Distributions.Truncated, bounds::Interval)
    # Note: Assumes that dist is result of trunctate, not of truncate_dist_hard:
    untrunc_dist = dist.untruncated

    min_lo = minimum(untrunc_dist)
    max_hi = maximum(untrunc_dist)
    lo = clamp(max(minimum(bounds), dist.lower), min_lo, max_hi)
    hi = clamp(max(lo, min(maximum(bounds), dist.upper)), min_lo, max_hi)
    trunc_dist = truncated(untrunc_dist, lo, hi)
    logrenormf = trunc_logpdf_ratio(dist, trunc_dist)
    return (dist = trunc_dist, logrenormf = logrenormf)
end


function truncate_dist_hard(d::Product, bounds::AbstractArray{<:Interval})
    @argcheck length(eachindex(bounds)) == length(d)
    r = truncate_dist_hard.(d.v, bounds)
    trunc_dists = map(x -> x.dist, r)
    logrenormf = sum(x.logrenormf for x in r)

    return (dist = Product(trunc_dists), logrenormf = logrenormf)
end


function truncate_dist_hard(d::StandardMvUniform, bounds::AbstractArray{<:Interval})
    @argcheck length(eachindex(bounds)) == length(d)
    n = length(eachindex(bounds))
    pd = product_distribution(Uniform.(fill(false, n), fill(true, n)))
    return truncate_dist_hard(pd, bounds)
end


function truncate_dist_hard(dist::ConstValueDist, bounds::AbstractVector{<:Interval})
    @argcheck length(eachindex(bounds)) == 0
    (dist = dist, logrenormf = 0)
end


function truncate_dist_hard(dist::NamedTupleDist{names,DT,AT,VT}, bounds::AbstractArray{<:Interval}) where {names,DT,AT,VT}
    @argcheck length(eachindex(bounds)) == totalndof(varshape(dist))
    distributions = values(dist)
    accessors = values(varshape(dist))

    r = map((dist, acc) -> truncate_dist_hard(dist, view(bounds, ValueShapes.view_idxs(eachindex(bounds), acc))), distributions, accessors)
    trunc_dist = NamedTupleDist(VT, NamedTuple{names}(map(x -> x.dist, r)))
    logrenormf = sum(map(x -> x.logrenormf, r))
    (dist = trunc_dist, logrenormf = logrenormf)
end

function truncate_dist_hard(dist::ValueShapes.UnshapedNTD, bounds::AbstractArray{<:Interval}) where names
    @argcheck length(eachindex(bounds)) == length(dist)
    r = truncate_dist_hard(dist.shaped, bounds)
    (dist = unshaped(r.dist), logrenormf = r.logrenormf)
end


"""
    BAT.trunc_logpdf_ratio(orig_dist::Distribution{TP}, trunc_dist::Distribution{TP})::AbstractFloat

*BAT-internal, not part of stable public API.*

Computes the log-ratio between the amplitude of the PDF of a truncated
distribution and the original (untruncted) distribution, within the support
of the truncated one.

The PDF of both distributions must have the same shape within the support of
`trunc_dist` and may only differ in amplitude.

Mainly used to implement [`BAT.truncate_density`](@ref), in conjunction with
[`BAT.truncate_dist_hard`](@ref). The result contributes to the `logrenormf`
factor of a [`RenormalizedDensity`] that uses truncated distributions internally,
to ensure the density does not get renormalized.
"""
function trunc_logpdf_ratio end


function trunc_logpdf_ratio(orig_dist::Distribution, trunc_dist::Distribution)
    _trunc_logpdf_ratio_fallback(orig_dist, trunc_dist)
end

function _trunc_logpdf_ratio_fallback(orig_dist::Distribution, trunc_dist::Distribution)
    x = rand(bat_determ_rng(), trunc_dist)
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
