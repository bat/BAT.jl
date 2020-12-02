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


function eval_logval(
    density::AbstractPosteriorDensity,
    v::Any,
    T::Type{<:Real} = density_logval_type(v);
    use_bounds::Bool = true,
    strict::Bool = false
)
    v_shaped = reshape_variate(varshape(density), v)
    if use_bounds && !variate_is_inbounds(density, v_shaped, strict)
        return log_zero_density(T)
    end

    prior_logval = eval_logval(
        getprior(density), v_shaped,
        use_bounds = false, strict = false
    )

    # Don't evaluate likelihood if prior probability is zero. Prevents
    # failures when algorithms try to explore parameter space outside of
    # definition of likelihood (as long as prior is chosen correctly).
    if !is_log_zero(prior_logval, T)
        likelihood_logval = eval_logval(
            getlikelihood(density), v_shaped,
            use_bounds = false, strict = false
        )
        convert(T, likelihood_logval + prior_logval)
    else
        log_zero_density(T)
    end
end


function eval_logval_unchecked(density::AbstractPosteriorDensity, v::Any)
    eval_logval(
        density, v,
        use_bounds = false, strict = false
    )
end


function var_bounds(density::AbstractPosteriorDensity)
    li_bounds = var_bounds(getlikelihood(density))
    pr_bounds = var_bounds(getprior(density))
    if ismissing(li_bounds)
        pr_bounds
    else
        li_bounds ∩ pr_bounds
    end
end


function estimate_finite_bounds(posterior::AbstractPosteriorDensity; bounds_type::BoundsType = hard_bounds)
    return estimate_finite_bounds(getprior(posterior), bounds_type = bounds_type)
end



@doc doc"""
    PosteriorDensity{
        Li<:AbstractDensity,
        Pr<:DistLikeDensity,
        ...
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
    P<:AbstractDensity,
    B<:AbstractVarBounds,
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
        var_bounds(li),
        var_bounds(pr)
    )

    parshapes = _posterior_parshapes(
        varshape(li),
        varshape(pr)
    )

    PosteriorDensity(li, pr, parbounds, parshapes)
end


getlikelihood(posterior::PosteriorDensity) = posterior.likelihood

getprior(posterior::PosteriorDensity) = posterior.prior

var_bounds(posterior::PosteriorDensity) = posterior.parbounds

ValueShapes.varshape(posterior::PosteriorDensity) = posterior.parshapes


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


_posterior_parbounds(li_bounds::AbstractVarBounds, pr_bounds::AbstractVarBounds) =
     li_bounds ∩ pr_bounds

_posterior_parbounds(li_bounds::Missing, pr_bounds::AbstractVarBounds) = pr_bounds
