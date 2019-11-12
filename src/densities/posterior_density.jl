# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    abstract type AbstractPosteriorDensity <: AbstractDensity end

Abstract super-type for posterior probability densities.
"""
abstract type AbstractPosteriorDensity <: AbstractDensity end
export AbstractPosteriorDensity


@doc doc"""
    getlikelihood(posterior::AbstractPosteriorDensity)::AbstractDensity

*BAT-internal, not part of stable public API.*

The likelihood density of `posterior`. The likelihood may or may not be
normalized.
"""
function getlikelihood end


@doc doc"""
    getprior(posterior::AbstractPosteriorDensity)::AbstractDensity

*BAT-internal, not part of stable public API.*

The prior density of `posterior`. The prior may or may not be normalized.
"""
function getprior end


@doc doc"""
    BAT.apply_bounds_and_eval_posterior_logval!(
        T::Type{<:Real},
        density::AbstractDensity,
        params::AbstractVector{<:Real}
    )

*BAT-internal, not part of stable public API.*

First apply bounds to the parameters, then compute and return posterior log
value. May modify `params`.

Guarantees that  :

* If parameters are still out of bounds after applying bounds,
  `density_logval` is not called for either prior or likelihood. 
* If `density_logval` for prior returns `-Inf`, `density_logval` is not called
  for likelihood.

In both cases, `T(-Inf)` is returned for both prior and posterior.
"""
function apply_bounds_and_eval_posterior_logval! end

function apply_bounds_and_eval_posterior_logval!(
    T::Type{<:Real},
    posterior::AbstractPosteriorDensity,
    params::AbstractVector{<:Real}
)
    bounds = param_bounds(posterior)
    apply_bounds!(params, bounds)

    parshapes = params_shape(posterior)
    zero_prob_logval = convert(T, -Inf)

    prior_logval = if !isoob(params)
        convert(T, eval_density_logval(getprior(posterior), params, parshapes))
    else
        zero_prob_logval
    end

    likelihood_logval = if prior_logval > zero_prob_logval
        convert(T, eval_density_logval(getlikelihood(posterior), params, parshapes))
    else
        zero_prob_logval
    end

    convert(T, prior_logval + likelihood_logval)
end


@doc doc"""
    BAT.apply_bounds_and_eval_posterior_logval_strict!(
        posterior::AbstractPosteriorDensity,
        params::AbstractVector{<:Real}
    )

*BAT-internal, not part of stable public API.*

First apply bounds to the parameters, then compute and return posterior log
value. May modify `params`.

Guarantees that  :

* If parameters are still out of bounds after applying bounds,
  `density_logval` is not called for either prior or likelihood. 
* If `density_logval` for prior returns `-Inf`, `density_logval` is not called
  for likelihood.

In both cases, an exception is thrown.
"""
function apply_bounds_and_eval_posterior_logval_strict!(
    posterior::AbstractPosteriorDensity,
    params::AbstractVector{<:Real}
)
    bounds = param_bounds(posterior)
    apply_bounds!(params, bounds)

    parshapes = params_shape(posterior)

    prior_logval = if !isoob(params)
        eval_density_logval(getprior(posterior), params, parshapes)
    else
        throw(ArgumentError("Parameter(s) out of bounds"))
    end

    likelihood_logval = if prior_logval > convert(typeof(prior_logval), -Inf)
        eval_density_logval(getlikelihood(posterior), params, parshapes)
    else
        throw(ErrorException("Prior density must not be zero."))
    end

    T = typeof(likelihood_logval)

    convert(T, convert(T, prior_logval) + likelihood_logval)
end


function density_logval(density::AbstractPosteriorDensity, params::AbstractVector{<:Real})
    parshapes = params_shape(density)
    eval_density_logval(getprior(density), params, parshapes) +
    eval_density_logval(getlikelihood(density), params, parshapes)
end

function density_logval(density::AbstractPosteriorDensity, params::Any)
    density_logval(getprior(density), params) +
    density_logval(getlikelihood(density), params)
end


function param_bounds(density::AbstractPosteriorDensity)
    li_bounds = param_bounds(getlikelihood(density))
    pr_bounds = param_bounds(getprior(density))
    if ismissing(li_bounds)
        pr_bounds
    else
        li_bounds ∩ pr_bounds
    end
end


function nparams(density::AbstractPosteriorDensity)
    li_np = nparams(getlikelihood(density))
    pr_np = nparams(getprior(density))
    ismissing(li_np) || li_np == pr_np || error("Likelihood and prior have different number of parameters")
    pr_np
end



@doc doc"""
    PosteriorDensity{
        Li<:AbstractDensity,
        Pr<:DistLikeDensity
    } <: AbstractPosteriorDensity

A representation of a PosteriorDensity, based a likelihood and prior, a
representation of the posterior density is cached internally. The densities
be accessed via

```julia
getlikelihood(posterior::PosteriorDensity)::Li
getprior(posterior::PosteriorDensity)::Pr
```

Constructors:

```julia
PosteriorDensity(likelihood::AbstractDensity, prior::AbstractDensity)
PosteriorDensity(likelihood::Any, prior::Any)
PosteriorDensity(log_likelihood::Function, prior::Any)
```
"""
struct PosteriorDensity{
    L<:AbstractDensity,
    P<:DistLikeDensity,
    B<:AbstractParamBounds,
    S<:AbstractValueShape
} <: AbstractPosteriorDensity
    likelihood::L
    prior::P
    parbounds::B
    parshapes::S
end

export PosteriorDensity

function PosteriorDensity(likelihood::Any, prior::Any)
    li = convert(AbstractDensity, likelihood)
    pr = convert(AbstractDensity, prior)

    parbounds = _posterior_parbounds(
        param_bounds(li),
        param_bounds(pr)
    )

    parshapes = _posterior_parshapes(
        params_shape(li),
        params_shape(pr)
    )

    PosteriorDensity(li, pr, parbounds, parshapes)
end


getlikelihood(posterior::PosteriorDensity) = posterior.likelihood

getprior(posterior::PosteriorDensity) = posterior.prior

param_bounds(posterior::PosteriorDensity) = posterior.parbounds

params_shape(posterior::PosteriorDensity) = posterior.parshapes


function _posterior_parshapes(li_ps::AbstractValueShape, pr_ps::AbstractValueShape)
    if li_ps == pr_ps
        li_ps
    else
        throw(ArgumentError("Variable shapes of likelihood and prior are incompatible"))
    end
end

function _posterior_parshapes(li_ps::AbstractValueShape, pr_ps::ArrayShape{T,1}) where T
    n = totalndof(li_ps)
    if n == totalndof(pr_ps)
        @assert size(pr_ps) == (n,)
        li_ps
    else
        throw(ArgumentError("Likelihood and prior have different number of free parameters"))
    end
end

_posterior_parshapes(li_ps::ArrayShape{T,1}, pr_ps::AbstractValueShape) where T =
    _posterior_parshapes(pr_ps, li_ps)

function _posterior_parshapes(li_ps::ArrayShape{T,1}, pr_ps::ArrayShape{U,1}) where {T,U}
    if size(li_ps) == size(pr_ps)
        li_ps
    else
        throw(ArgumentError("Likelihood and prior have different number of free parameters"))
    end
end

_posterior_parshapes(li_ps::Missing, pr_ps::AbstractValueShape) = pr_ps

_posterior_parshapes(li_ps::Union{AbstractValueShape,Missing}, pr_ps::Missing) =
    throw(ArgumentError("Parameter shapes of prior must not be missing"))


_posterior_parbounds(li_bounds::AbstractParamBounds, pr_bounds::AbstractParamBounds) =
     li_bounds ∩ pr_bounds

_posterior_parbounds(li_bounds::Missing, pr_bounds::AbstractParamBounds) = pr_bounds
