# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    lbqintegral(integrand, base)

The Lebesgue integral of `integrand` over `base`. `base` must
be (convertible to) a `BATMeasure` and `integrand` must be
(convertible to) a `BATDensity`.
"""
function lbqintegral end
export lbqintegral


"""
    abstract type AbstractPosteriorMeasure <: BATMeasure end

Abstract type for posterior probability densities.
"""
abstract type AbstractPosteriorMeasure <: BATMeasure end
export AbstractPosteriorMeasure

MeasureBase.getdof(nu::AbstractPosteriorMeasure) = MeasureBase.getdof(basemeasure(nu))


"""
    getintegrand(nu::PosteriorMeasure)
    getintegrand(nu::MeasureBase.DensityMeasure)

*BAT-internal, not part of stable public API.*
"""
function getintegrand end

BAT.getintegrand(nu::MeasureBase.DensityMeasure) = nu.f


"""
    getlikelihood(posterior::AbstractPosteriorMeasure)::BATDensity

*BAT-internal, not part of stable public API.*

The likelihood density of `posterior`. The likelihood may or may not be
normalized.
"""
function getlikelihood end

getlikelihood(nu::MeasureBase.DensityMeasure) = getintegrand(nu)


"""
    getprior(posterior::AbstractPosteriorMeasure)::BATMeasure

*BAT-internal, not part of stable public API.*

The prior density of `posterior`. The prior may or may not be normalized.
"""
function getprior end

getprior(nu::MeasureBase.DensityMeasure) = basemeasure(nu)


function BAT.logdensityof_batmeasure(density::AbstractPosteriorMeasure, v::Any)
    R = density_valtype(density, v)

    prior_logval = logdensityof(basemeasure(density), v)

    # Don't evaluate integrand if basemeasure probability is zero. Prevents
    # failures when algorithms try to explore parameter space outside of
    # definition of integrand (as long as basemeasure is chosen correctly).
    if !is_log_zero(prior_logval, R)
        likelihood_logval = logdensityof(integrand(density), v)
        convert(R, likelihood_logval + prior_logval)::R
    else
        log_zero_density(R)::R
    end
end


"""
    struct PosteriorMeasure

A representation of a PosteriorMeasure, based a likelihood and prior.
Likelihood and prior be accessed via

```julia
getlikelihood(posterior::PosteriorMeasure)::Li
getprior(posterior::PosteriorMeasure)::Pr
```
"""
struct PosteriorMeasure{
    L<:BATDensity,
    P<:BATMeasure,
} <: AbstractPosteriorMeasure
    density::L
    basemeasure::P
end

#!!!!!!!!!
export PosteriorMeasure

BAT.getintegrand(nu::PosteriorMeasure) = nu.getintegrand
MeasureBase.basemeasure(nu::PosteriorMeasure) = nu.basemeasure

getlikelihood(nu::PosteriorMeasure) = getintegrand(nu)
getprior(nu::PosteriorMeasure) = basemeasure(nu)

ValueShapes.varshape(nu::PosteriorMeasure) = varshape(basemeasure(nu))


function ValueShapes.unshaped(density::PosteriorMeasure)
    f_pullback = varshape(density.prior)
    lbqintegral(getintegrand(density) ∘ f_pullback, unshaped(basemeasure(density)))
end

function (shape::AbstractValueShape)(density::PosteriorMeasure)
    f_pushforward = inverse(shape)
    lbqintegral(getintegrand(density) ∘ f_pushforward, shape(basemeasure(density)))
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
