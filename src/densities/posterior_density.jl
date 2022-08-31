# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractPosteriorMeasure <: BATMeasure end

Abstract type for posterior probability densities.
"""
abstract type AbstractPosteriorMeasure <: BATMeasure end
export AbstractPosteriorMeasure


"""
    getlikelihood(posterior::AbstractPosteriorMeasure)::BATDenstiy

*BAT-internal, not part of stable public API.*

The likelihood density of `posterior`. The likelihood may or may not be
normalized.
"""
function getlikelihood end


"""
    getprior(posterior::AbstractPosteriorMeasure)::BATMeasure

*BAT-internal, not part of stable public API.*

The prior density of `posterior`. The prior may or may not be normalized.
"""
function getprior end


function DensityInterface.logdensityof(density::AbstractPosteriorMeasure, v::Any)
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


function checked_logdensityof(density::AbstractPosteriorMeasure, v::Any)
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



function var_bounds(density::AbstractPosteriorMeasure)
    li_bounds = var_bounds(getlikelihood(density))
    pr_bounds = var_bounds(getprior(density))
    if ismissing(li_bounds)
        pr_bounds
    else
        li_bounds ∩ pr_bounds
    end
end



"""
    struct PosteriorMeasure{
        Li<:AbstractMeasureOrDensity,
        Pr<:DistLikeMeasure,
        ...
    } <: AbstractPosteriorMeasure

A representation of a PosteriorMeasure, based a likelihood and prior.
Likelihood and prior be accessed via

```julia
getlikelihood(posterior::PosteriorMeasure)::Li
getprior(posterior::PosteriorMeasure)::Pr
```

Constructors:

* ```PosteriorMeasure(likelihood, prior)```
* ```PosteriorMeasure{T<:Real}(likelihood, prior)```

`likelihood` and `prior` must be convertible to an [`AbstractMeasureOrDensity`](@ref).

Fields:

$(TYPEDFIELDS)

!!! note

    Fields `parbounds` and `parbounds` do not form part of the stable public
    API and are subject to change without deprecation.
"""
struct PosteriorMeasure{
    VT<:Real,
    DT<:Real,
    L<:AbstractMeasureOrDensity,
    P<:AbstractMeasureOrDensity,
    S<:AbstractValueShape,
    B<:AbstractVarBounds,
} <: AbstractPosteriorMeasure
    likelihood::L
    prior::P
    parshapes::S
    parbounds::B
end

export PosteriorMeasure


function PosteriorMeasure{VT,DT}(
    likelihood::AbstractMeasureOrDensity, prior::AbstractMeasureOrDensity, parshapes::AbstractValueShape, parbounds::AbstractVarBounds
) where {VT<:Real,DT<:Real}
    @argcheck DensityKind(likelihood) isa IsDensity
    @argcheck DensityKind(prior) isa HasDensity

    L = typeof(likelihood); P = typeof(prior);
    S = typeof(parshapes); B = typeof(parbounds);
    PosteriorMeasure{VT,DT,L,P,S,B}(likelihood, prior, parshapes, parbounds)
end


PosteriorMeasure(μ::DensityMeasure) = PosteriorMeasure(μ.f, μ.base)
Base.convert(::Type{AbstractMeasureOrDensity}, μ::DensityMeasure) = PosteriorMeasure(μ)


function _preproc_likelihood_prior(likelihood::Any, prior::Any)
    li = convert(AbstractMeasureOrDensity, likelihood)
    pr = convert(AbstractMeasureOrDensity, prior)

    parbounds = _posterior_parbounds(var_bounds(li), var_bounds(pr))

    li_shape = varshape(li)
    pr_shape = varshape(pr)
    parshapes = _posterior_parshapes(li_shape, pr_shape)

    li_with_shape = _density_with_shape(li, parshapes, li_shape)
    li_with_shape, pr, parshapes, parbounds
end


function PosteriorMeasure{VT,DT}(likelihood::Any, prior::Any) where {VT<:Real,DT<:Real}
    li, pr, parshapes, parbounds = _preproc_likelihood_prior(likelihood, prior)
    PosteriorMeasure{VT,DT}(li, pr, parshapes, parbounds)
end

function PosteriorMeasure{VT}(likelihood::Any, prior::Any) where {VT<:Real}
    li, pr, parshapes, parbounds = _preproc_likelihood_prior(likelihood, prior)
    DT = default_val_numtype(li)
    PosteriorMeasure{VT,DT}(li, pr, parshapes, parbounds)
end

function PosteriorMeasure(likelihood::Any, prior::Any)
    li, pr, parshapes, parbounds = _preproc_likelihood_prior(likelihood, prior)
    VT = default_val_numtype(li)
    DT = default_val_numtype(li)
    PosteriorMeasure{VT,DT}(li, pr, parshapes, parbounds)
end


getlikelihood(posterior::PosteriorMeasure) = posterior.likelihood

getprior(posterior::PosteriorMeasure) = posterior.prior

var_bounds(posterior::PosteriorMeasure) = posterior.parbounds

ValueShapes.varshape(posterior::PosteriorMeasure) = posterior.parshapes

ValueShapes.unshaped(density::PosteriorMeasure) = PosteriorMeasure(unshaped(density.likelihood), unshaped(density.prior))

(shape::AbstractValueShape)(density::PosteriorMeasure) = PosteriorMeasure(shape(density.likelihood), shape(density.prior))


function _posterior_parshapes(li_ps::AbstractValueShape, pr_ps::AbstractValueShape)
    if li_ps == pr_ps
        li_ps
    else
        throw(ArgumentError("Variable shapes of likelihood and prior are incompatible"))
    end
end

function _posterior_parshapes(li_ps::AbstractValueShape, pr_ps::ArrayShape{T,1}) where T
    throw(ArgumentError("Variable shapes of likelihood and prior are incompatible"))
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


function _density_with_shape(density::AbstractMeasureOrDensity, requested_shape::AbstractValueShape, orig_shape::AbstractValueShape)
    if requested_shape == orig_shape
        density
    else
        throw(ArgumentError("Original and requested variable shape are incompatible"))
    end
end

function _density_with_shape(density::AbstractMeasureOrDensity, requested_shape::AbstractValueShape, orig_shape::Missing)
    DensityWithShape(density, requested_shape)
end


function example_posterior()
    rng = StableRNGs.StableRNG(0x4cf83495c736cac2)
    prior = NamedTupleDist(
        b = [4.2, 3.3],
        a = Exponential(),
        c = Normal(1, 3),
        d = [Weibull(), Weibull()],
        e = Beta(),
        f = MvNormal([0.3, -2.9], [1.7 0.5; 0.5 2.3])
    )
    n = totalndof(varshape(prior))
    A = randn(rng, n, n)
    likelihood = logfuncdensity(logdensityof(varshape(prior)(MvNormal(A * A'))))
    PosteriorMeasure(likelihood, prior)
end


function example_posterior_with_dirichlet()
    rng = StableRNGs.StableRNG(0x4cf83495c736cac2)
    prior = merge(BAT.example_posterior().prior.dist, (g = Dirichlet([1.2, 2.4, 3.6]),))
    n = totalndof(varshape(prior))
    A = randn(rng, n, n)
    likelihood = logfuncdensity(logdensityof(varshape(prior)(MvNormal(A * A'))))
    PosteriorMeasure(likelihood, prior)
end
