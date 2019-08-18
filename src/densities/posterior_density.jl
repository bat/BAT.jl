# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractPosteriorDensity <: AbstractDensity end


doc"""
    getlikelihood(posterior::AbstractPosteriorDensity)::AbstractDensity

The likelihood density of `posterior`. The likelihood may or may not be
normalized.
"""
function getlikelihood end
export getlikelihood


doc"""
    getprior(posterior::AbstractPosteriorDensity)::AbstractDensity

The prior density of `posterior`. The prior may or may not be normalized.
"""
function getprior end
export getprior


@doc """
    BAT.eval_prior_posterior_logval!(
        T::Type{<:Real},
        density::AbstractDensity,
        params::AbstractVector{<:Real}
    )

First apply bounds to the parameters, compute prior and posterior log values
by via `eval_density_logval`.

May modify `params`.

Returns a `NamedTuple{(:log_prior, :log_posterior),Tuple{T,T}}`

Guarantees that  :

* If parameters are still out of bounds after applying bounds,
  `density_logval` is not called for either prior or likelihood. 
* If `density_logval` for prior returns `-Inf`, `density_logval` is not called
  for likelihood.

In both cases, `T(-Inf)` is returned for both prior and posterior.
"""
function eval_prior_posterior_logval! end

function eval_prior_posterior_logval!(
    T::Type{<:Real},
    posterior::AbstractPosteriorDensity,
    params::AbstractVector{<:Real}
)
    bounds = param_bounds(posterior)
    apply_bounds!(params, bounds)

    parshapes = param_shapes(posterior)
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

    posterior_logval = convert(T, prior_logval + likelihood_logval)

    (log_prior = prior_logval, log_posterior = posterior_logval)
end


@doc """
    BAT.eval_prior_posterior_logval_strict!(
        density::AbstractDensity,
        params::AbstractVector{<:Real}
    )

First apply bounds to the parameters, compute prior and posterior log values
by via `eval_density_logval`.

May modify `params`.

Returns a `NamedTuple{(:log_prior, :log_posterior),Tuple{T,T}}`. T is
inferred from value returned by `eval_density_logval` for the likelihood.

Guarantees that  :

* If parameters are still out of bounds after applying bounds,
  `density_logval` is not called for either prior or likelihood. 
* If `density_logval` for prior returns `-Inf`, `density_logval` is not called
  for likelihood.

In both cases, an exception is thrown.
"""
function eval_prior_posterior_logval_strict!(
    posterior::AbstractPosteriorDensity,
    params::AbstractVector{<:Real}
)
    bounds = param_bounds(posterior)
    apply_bounds!(params, bounds)

    parshapes = param_shapes(posterior)

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

    posterior_logval = convert(T, convert(T, prior_logval) + likelihood_logval)

    (log_prior = prior_logval, log_posterior = posterior_logval)
end


function density_logval(density::AbstractPosteriorDensity, params::AbstractVector{<:Real})
    parshapes = param_shapes(density)
    eval_density_logval(getprior(density), params, parshapes) +
    eval_density_logval(getlikelihood(density), params, parshapes)
end


function density_logval(density::AbstractPosteriorDensity, params::NamedTuple)
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



doc"""

    PosteriorDensity{
        Li<:AbstractDensity,
        Pr<:AbstractPriorDensity
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
    P<:AbstractPriorDensity,
    B<:AbstractParamBounds,
    S<:Union{VarShapes,Nothing}
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
        param_shapes(li),
        param_shapes(pr)
    )

    PosteriorDensity(li, pr, parbounds, parshapes)
end


getlikelihood(posterior::PosteriorDensity) = posterior.likelihood

getprior(posterior::PosteriorDensity) = posterior.prior

param_bounds(posterior::PosteriorDensity) = posterior.parbounds

param_shapes(posterior::PosteriorDensity) = posterior.parshapes


function _posterior_parshapes(li_ps::Union{VarShapes,Nothing}, pr_ps::Union{VarShapes,Nothing})
    pr_ps == li_ps || throw(ArgumentError("Variable shapes of likelihood and prior do not match"))
    li_ps
end

_posterior_parshapes(li_ps::Missing, pr_ps::Union{VarShapes,Nothing}) = pr_ps

_posterior_parshapes(li_ps::Union{VarShapes,Nothing,Missing}, pr_ps::Missing) =
    throw(ArgumentError("Parameter shapes of prior must not be missing"))


_posterior_parbounds(li_bounds::AbstractParamBounds, pr_bounds::AbstractParamBounds) =
     li_bounds ∩ pr_bounds

_posterior_parbounds(li_bounds::Missing, pr_bounds::AbstractParamBounds) = pr_bounds
