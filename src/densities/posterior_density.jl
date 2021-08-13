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


function DensityInterface.logdensityof(density::AbstractPosteriorDensity, v::Any)
    R = density_valtype(density, v)

    prior_logval = logdensityof(getprior(density), v)

    # Don't evaluate likelihood if prior probability is zero. Prevents
    # failures when algorithms try to explore parameter space outside of
    # definition of likelihood (as long as prior is chosen correctly).
    if !is_log_zero(prior_logval, R)
        likelihood_logval = logdensityof(getlikelihood(density), v)
        convert(R, likelihood_logval + prior_logval)::R
    else
        log_zero_density(R)::R
    end
end


function checked_logdensityof(density::AbstractPosteriorDensity, v::Any)
    R = density_valtype(density, v)

    prior_logval = checked_logdensityof(getprior(density), v)

    # Don't evaluate likelihood if prior probability is zero. Prevents
    # failures when algorithms try to explore parameter space outside of
    # definition of likelihood (as long as prior is chosen correctly).
    if !is_log_zero(prior_logval, R)
        likelihood_logval = checked_logdensityof(getlikelihood(density), v)
        convert(R, likelihood_logval + prior_logval)::R
    else
        log_zero_density(R)::R
    end
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
* ```PosteriorDensity{T<:Real}(likelihood, prior)```

`likelihood` and `prior` must be convertible to an [`AbstractDensity`](@ref).

Fields:

$(TYPEDFIELDS)

!!! note

    Fields `parbounds` and `parbounds` do not form part of the stable public
    API and are subject to change without deprecation.
"""
struct PosteriorDensity{
    VT<:Real,
    DT<:Real,
    L<:AbstractDensity,
    P<:AbstractDensity,
    S<:AbstractValueShape,
    B<:AbstractVarBounds,
} <: AbstractPosteriorDensity
    likelihood::L
    prior::P
    parshapes::S
    parbounds::B
end

export PosteriorDensity


function PosteriorDensity{VT,DT}(
    likelihood::AbstractDensity, prior::AbstractDensity, parshapes::AbstractValueShape, parbounds::AbstractVarBounds
) where {VT<:Real,DT<:Real}
    L = typeof(likelihood); P = typeof(prior);
    S = typeof(parshapes); B = typeof(parbounds);
    PosteriorDensity{VT,DT,L,P,S,B}(likelihood, prior, parshapes, parbounds)
end


function _preproc_likelihood_prior(likelihood::Any, prior::Any)
    li = convert(AbstractDensity, likelihood)
    pr = convert(AbstractDensity, prior)

    parbounds = _posterior_parbounds(var_bounds(li), var_bounds(pr))

    li_shape = varshape(li)
    pr_shape = varshape(pr)
    parshapes = _posterior_parshapes(li_shape, pr_shape)

    li_with_shape = _density_with_shape(li, parshapes, li_shape)
    li_with_shape, pr, parshapes, parbounds
end


function PosteriorDensity{VT,DT}(likelihood::Any, prior::Any) where {VT<:Real,DT<:Real}
    li, pr, parshapes, parbounds = _preproc_likelihood_prior(likelihood, prior)
    PosteriorDensity{VT,DT}(li, pr, parshapes, parbounds)
end

function PosteriorDensity{VT}(likelihood::Any, prior::Any) where {VT<:Real}
    li, pr, parshapes, parbounds = _preproc_likelihood_prior(likelihood, prior)
    DT = default_val_numtype(li)
    PosteriorDensity{VT,DT}(li, pr, parshapes, parbounds)
end

function PosteriorDensity(likelihood::Any, prior::Any)
    li, pr, parshapes, parbounds = _preproc_likelihood_prior(likelihood, prior)
    VT = default_val_numtype(li)
    DT = default_val_numtype(li)
    PosteriorDensity{VT,DT}(li, pr, parshapes, parbounds)
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


function example_posterior_with_dirichlet()
    prior = merge(BAT.example_posterior().prior.dist, (g = Dirichlet([1.2, 2.4, 3.6]),))
    n = totalndof(varshape(prior))
    likelihood = varshape(prior)(MvNormal(float(I(n))))
    PosteriorDensity(likelihood, prior)
end
