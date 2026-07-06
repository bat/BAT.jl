# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractPosteriorMeasure <: BATMeasure end

Abstract type for posterior probability densities.
"""
abstract type AbstractPosteriorMeasure <: BATMeasure end
export AbstractPosteriorMeasure

MeasureBase.basemeasure(m::AbstractPosteriorMeasure) = MeasureBase.basemeasure(getprior(m))
MeasureBase.getdof(m::AbstractPosteriorMeasure) = MeasureBase.getdof(getprior(m))

function _bat_weightedmeasure(logweight::Real, m::AbstractPosteriorMeasure)
    likelihood, prior = getlikelihood(m), getprior(m)
    new_likelihood = logfuncdensity(ffcomp(Base.Fix2(+, logweight), logdensityof(likelihood)))
    lbqintegral(new_likelihood, prior)
end



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
    likelihood, prior = getlikelihood(density), getprior(density)

    raw_prior_logval = logdensityof(prior, v)

    T = typeof(raw_prior_logval)
    U = density_valtype(likelihood, v)
    R = promote_type(T, U)

    prior_logval = convert_density_value(R, raw_prior_logval)

    # Don't evaluate likelihood if prior probability is zero. Prevents
    # failures when algorithms try to explore parameter space outside of
    # definition of likelihood (as long as prior is chosen correctly).
    if !is_log_zero(prior_logval, R)
        likelihood_logval = logdensityof(getlikelihood(density), v)
        convert_density_value(R, likelihood_logval + prior_logval)
    else
        convert_density_value(R, log_zero_density(T))
    end
end


function checked_logdensityof(density::AbstractPosteriorMeasure, v::Any)
    likelihood, prior = getlikelihood(density), getprior(density)

    raw_prior_logval = checked_logdensityof(prior, v)

    T = typeof(raw_prior_logval)
    U = density_valtype(likelihood, v)
    R = promote_type(T, U)

    prior_logval = convert_density_value(R, raw_prior_logval)

    # Don't evaluate likelihood if prior probability is zero. Prevents
    # failures when algorithms try to explore parameter space outside of
    # definition of likelihood (as long as prior is chosen correctly).
    if !is_log_zero(prior_logval, R)
        likelihood_logval = checked_logdensityof(getlikelihood(density), v)
        convert_density_value(R, likelihood_logval + prior_logval)
    else
        convert_density_value(R, log_zero_density(T))
    end
end


"""
    struct PosteriorMeasure{Li,Pr<:AbstractMeasure} <: AbstractPosteriorMeasure

A representation of a PosteriorMeasure, based a likelihood and prior.
Likelihood and prior be accessed via

```julia
getlikelihood(posterior::PosteriorMeasure)::Li
getprior(posterior::PosteriorMeasure)::Pr
```

Constructors:

* ```PosteriorMeasure(likelihood, prior)```
* ```PosteriorMeasure{T<:Real}(likelihood, prior)```

Fields:

$(TYPEDFIELDS)
"""
struct PosteriorMeasure{L,P<:AbstractMeasure} <: AbstractPosteriorMeasure
    likelihood::L
    prior::P
end

export PosteriorMeasure

_convert_likelihood(likelihood, ::IsDensity) = likelihood
_convert_likelihood(::Any, ::HasDensity) = throw(ArgumentError("Likelihood must be a density, not like a measure that has a density."))
_convert_likelihood(f_likelihood, ::NoDensity) = logfuncdensity(ffcomp(logvalof, f_likelihood))

function PosteriorMeasure(
    likelihood::Any, prior::Union{AbstractMeasure,Distribution}
)
    li = _convert_likelihood(likelihood, DensityKind(likelihood))
    pr = batmeasure(prior)
    L = typeof(li); P = typeof(pr);
    PosteriorMeasure{L,P}(li, pr)
end


PosteriorMeasure(μ::DensityMeasure) = PosteriorMeasure(μ.f, μ.base)
Base.convert(::Type{BATMeasure}, μ::DensityMeasure) = PosteriorMeasure(μ)


getlikelihood(posterior::PosteriorMeasure) = posterior.likelihood

getprior(posterior::PosteriorMeasure) = posterior.prior

measure_support(posterior::PosteriorMeasure) = measure_support(getprior(posterior))

ValueShapes.varshape(posterior::PosteriorMeasure) = varshape(getprior(posterior))

ValueShapes.unshaped(posterior::PosteriorMeasure) = _unshaped_with(posterior, varshape(posterior))

# ToDo: Check size:
_unshaped_with(posterior::PosteriorMeasure, ::ArrayShape{<:Real,1}) = posterior

function _unshaped_with(posterior::PosteriorMeasure, shp::AbstractValueShape)
    li, pr = getlikelihood(posterior), getprior(posterior)
    lbqintegral(_precompose_density(li, shp), unshaped(pr, shp))
end


(shape::AbstractValueShape)(density::PosteriorMeasure) = PosteriorMeasure(shape(density.likelihood), shape(density.prior))
