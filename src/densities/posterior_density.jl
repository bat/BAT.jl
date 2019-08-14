# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractPosteriorDensity <: AbstractDensity end


doc"""
    BAT.likelihood(posterior::AbstractPosteriorDensity)::AbstractDensity

The likelihood density of `posterior`. The likelihood may or may not be
normalized.
"""
function likelihood end


doc"""
    BAT.prior(posterior::AbstractPosteriorDensity)::AbstractDensity

The prior density of `posterior`. The prior may or may not be normalized.
"""
function prior end


@doc """
    BAT.eval_prior_posterior_logval!(...)

Internal function to first apply bounds to the parameters and then
compute prior and posterior log valued.
"""
function eval_prior_posterior_logval! end

function eval_prior_posterior_logval!(
    T::Type{<:Real},
    posterior::AbstractPosteriorDensity,
    params::AbstractVector{<:Real}
)
    bounds = param_bounds(posterior)
    apply_bounds!(params, bounds)

    prior_logval = eval_density_logval!(T, prior(posterior), params, do_applybounds = false)
    likelihood_logval = if prior_logval > -Inf
        eval_density_logval!(T, likelihood(posterior), params, do_applybounds = false)
    else
        T(-Inf)
    end
    posterior_logval = prior_logval + likelihood_logval
    @assert !isnan(prior_logval) || !isnan(posterior_logval)
    (prior_logval, posterior_logval)
end


function density_logval(density::AbstractPosteriorDensity, params::AbstractVector{<:Real})
    density_logval(likelihood(density)) + density_logval(prior(density))
end


function param_bounds(density::AbstractPosteriorDensity)
    li_bounds = param_bounds(likelihood(density))
    pr_bounds = param_bounds(prior(density))
    if ismissing(li_bounds)
        pr_bounds
    else
        li_bounds ∩ pr_bounds
    end
end


function nparams(density::AbstractPosteriorDensity)
    li_np = nparams(likelihood(density))
    pr_np = nparams(prior(density))
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
likelihood(posterior::PosteriorDensity)::Li
prior(posterior::PosteriorDensity)::Pr
```

Constructors:

```julia
PosteriorDensity(likelihood::AbstractDensity, prior::AbstractDensity)
PosteriorDensity(likelihood::Any, prior::Any)
PosteriorDensity(log_likelihood::Function, prior::Any)
```
"""
struct PosteriorDensity{
    Li<:AbstractDensity,
    Pr<:AbstractPriorDensity
} <: AbstractPosteriorDensity
    likelihood::Li
    prior::Pr

    function PosteriorDensity{Li,Pr}(
        likelihood::Li,
        prior::Pr
    ) where {
        Li<:AbstractDensity,
        Pr<:AbstractDensity
    }
        # ToDo: Check compatibility of likelihood and prior
        new(likelihood, prior)
    end
end

export PosteriorDensity


function PosteriorDensity(likelihood::AbstractDensity, prior::AbstractDensity)
    li = convert(AbstractDensity, likelihood)
    pr = convert(AbstractDensity, prior)
    Li = typeof(likelihood)
    Pr = typeof(prior)
    PosteriorDensity{Li,Pr}(li, pr)
end

function PosteriorDensity(likelihood::Any, prior::Any)
    conv_prior = convert(AbstractDensity, prior)
    conv_likelihood = convert(AbstractDensity, likelihood)
    PosteriorDensity(conv_likelihood, conv_prior)
end

function PosteriorDensity(log_likelihood::Function, prior::Any)
    conv_prior = convert(AbstractDensity, prior)
    conv_likelihood = GenericDensity(log_likelihood, nparams(conv_prior))
    PosteriorDensity(conv_likelihood, conv_prior)
end


likelihood(posterior::PosteriorDensity) = posterior.likelihood

prior(posterior::PosteriorDensity) = posterior.prior
