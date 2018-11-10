# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractBayesianModel end


doc"""
    BAT.likelihood(model::AbstractBayesianModel)::AbstractDensity

The likelihood density of `model`. The likelihood may or may not be
normalized.
"""
function likelihood end


doc"""
    BAT.prior(model::AbstractBayesianModel)::AbstractDensity

The prior density of `model`. The prior may or may not be normalized.
"""
function prior end


doc"""
    BAT.posterior(model::AbstractBayesianModel)

The posterior of `model`. Must be equivalent to
`likelihood(model) * prior(model)`.
"""
function posterior end


nparams(model::AbstractBayesianModel) = nparams(likelihood(model))



@doc """
    BAT.eval_prior_posterior_logval!(...)

Internal function to first apply bounds to the parameters and then
compute prior and posterior log valued.
"""
function eval_prior_posterior_logval! end

function eval_prior_posterior_logval!(
    T::Type{<:Real},
    model::AbstractBayesianModel,
    params::AbstractVector{<:Real},
    exec_context::ExecContext
)
    prior_logval = eval_density_logval!(T, prior(model), params, exec_context)
    likelihood_logval = if prior_logval > -Inf
        eval_density_logval!(T, likelihood(model), params, exec_context)
    else
        T(-Inf)
    end
    posterior_logval = prior_logval + likelihood_logval
    @assert !isnan(prior_logval) || !isnan(posterior_logval)
    (prior_logval, posterior_logval)
end



doc"""

    BayesianModel{Li,Pr,Po} <: AbstractBayesianModel

A representation of a BayesianModel, based a likelihood and prior, a
representation of the posterior density is cached internally. The densities
be accessed via

```julia
likelihood(model::BayesianModel)::Li
prior(model::BayesianModel)::Pr
posterior(model::BayesianModel)::Po
```

Constructors:

```julia
BayesianModel(likelihood::AbstractDensity, prior::AbstractDensity)
BayesianModel(likelihood::Any, prior::Any)
BayesianModel(log_likelihood::Function, prior::Any)
```
"""
struct BayesianModel{
    Li<:AbstractDensity,
    Pr<:AbstractDensity,
    Po<:AbstractDensity
} <: AbstractBayesianModel
    likelihood::Li
    prior::Pr
    posterior::Po

    function BayesianModel{Li,Pr}(
        likelihood::Li,
        prior::Pr
    ) where {
        Li<:AbstractDensity,
        Pr<:AbstractDensity
    }
        posterior = likelihood * prior

        Po = typeof(posterior)
        new{Li,Pr,Po}(likelihood, prior, posterior)
    end
end

export BayesianModel


function BayesianModel(likelihood::AbstractDensity, prior::AbstractDensity)
    li = convert(AbstractDensity, likelihood)
    pr = convert(AbstractDensity, prior)
    Li = typeof(likelihood)
    Pr = typeof(prior)
    BayesianModel{Li,Pr}(li, pr)
end

function BayesianModel(likelihood::Any, prior::Any)
    conv_prior = convert(AbstractDensity, prior)
    conv_likelihood = convert(AbstractDensity, likelihood)
    BayesianModel(conv_likelihood, conv_prior)
end

function BayesianModel(log_likelihood::Function, prior::Any)
    conv_prior = convert(AbstractDensity, prior)
    conv_likelihood = GenericDensity(log_likelihood, nparams(conv_prior))
    BayesianModel(conv_likelihood, conv_prior)
end


likelihood(model::BayesianModel) = model.likelihood

prior(model::BayesianModel) = model.prior

posterior(model::BayesianModel) = model.posterior
