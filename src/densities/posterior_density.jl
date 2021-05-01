# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractPosteriorDensity <: AbstractDensity end

Abstract type for posterior probability densities.
"""
abstract type AbstractPosteriorDensity <: AbstractDensity end
export AbstractPosteriorDensity


"""
    getlikelihood(posterior::AbstractPosteriorDensity)::AbstractDensity

*BAT-internal, not part of stable public API.*

The likelihood density of `posterior`. The likelihood may or may not be
normalized.
"""
function getlikelihood end


"""
    getprior(posterior::AbstractPosteriorDensity)::AbstractDensity

*BAT-internal, not part of stable public API.*

The prior density of `posterior`. The prior may or may not be normalized.
"""
function getprior end


function eval_logval(density::AbstractPosteriorDensity, v::Any, T::Type{<:Real})
    v_shaped = fixup_variate(varshape(density), v)
    R = density_logval_type(v_shaped, T)
    
    prior_logval = eval_logval(getprior(density), v_shaped, R)

    # Don't evaluate likelihood if prior probability is zero. Prevents
    # failures when algorithms try to explore parameter space outside of
    # definition of likelihood (as long as prior is chosen correctly).
    if !is_log_zero(prior_logval, R)
        likelihood_logval = eval_logval(getlikelihood(density), v_shaped, R)
        convert(R, likelihood_logval + prior_logval)
    else
        log_zero_density(R)
    end
end

eval_logval_unchecked(density::AbstractPosteriorDensity, v::Any) = eval_logval(density, v, default_dlt())


function var_bounds(density::AbstractPosteriorDensity)
    li_bounds = var_bounds(getlikelihood(density))
    pr_bounds = var_bounds(getprior(density))
    if ismissing(li_bounds)
        pr_bounds
    else
        li_bounds ∩ pr_bounds
    end
end



"""
    struct PosteriorDensity{
        Li<:AbstractDensity,
        Pr<:DistLikeDensity,
        ...
    } <: AbstractPosteriorDensity

A representation of a PosteriorDensity, based a likelihood and prior.
Likelihood and prior be accessed via

```julia
getlikelihood(posterior::PosteriorDensity)::Li
getprior(posterior::PosteriorDensity)::Pr
```

Constructors:

* ```PosteriorDensity(likelihood, prior)```

`likelihood` and `prior` must be convertible to an [`AbstractDensity`](@ref).

Fields:

$(TYPEDFIELDS)

!!! note

    Fields `parbounds` and `parbounds` do not form part of the stable public
    API and are subject to change without deprecation.
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

    parbounds = _posterior_parbounds(var_bounds(li), var_bounds(pr))

    li_shape = varshape(li)
    pr_shape = varshape(pr)
    parshapes = _posterior_parshapes(li_shape, pr_shape)

    li_with_shape = _density_with_shape(li, parshapes, li_shape)

    PosteriorDensity(li_with_shape, pr, parbounds, parshapes)
end


getlikelihood(posterior::PosteriorDensity) = posterior.likelihood

getprior(posterior::PosteriorDensity) = posterior.prior

var_bounds(posterior::PosteriorDensity) = posterior.parbounds

ValueShapes.varshape(posterior::PosteriorDensity) = posterior.parshapes

ValueShapes.unshaped(density::PosteriorDensity) = PosteriorDensity(unshaped(density.likelihood), unshaped(density.prior))

(shape::AbstractValueShape)(density::PosteriorDensity) = PosteriorDensity(shape(density.likelihood), shape(density.prior))


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


function _density_with_shape(density::AbstractDensity, requested_shape::AbstractValueShape, orig_shape::AbstractValueShape)
    if requested_shape == orig_shape
        density
    else
        throw(ArgumentError("Original and requested variable shape are incompatible"))
    end
end

function _density_with_shape(density::AbstractDensity, requested_shape::AbstractValueShape, orig_shape::Missing)
    DensityWithShape(density, requested_shape)
end


function example_posterior()
    prior = NamedTupleDist(
        a = Exponential(),
        b = [4.2, 3.3],
        c = Normal(1, 3),
        d = [Weibull(), Weibull()],
        e = Beta(),
        f = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3])
    )
    n = totalndof(varshape(prior))
    likelihood = varshape(prior)(MvNormal(float(I(n))))
    PosteriorDensity(likelihood, prior)
end
