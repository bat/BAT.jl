# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    WhiteningAlgorithm

Abstract type for integration algorithms.
"""
abstract type IntegrationAlgorithm end
export IntegrationAlgorithm


"""
    AHMIntegration

Adaptive Harmonic Mean Integration algoritm
(Caldwell et al.)[https://arxiv.org/abs/1808.08051].

Constructor:

AHMIntegration(;kwargs...)

Optional Parameters/settings (`kwargs`):

* `whitening::WhiteningAlgorithm = CholeskyPartialWhitening()`

* `autocorlen::AutocorLenAlgorithm = GeyerAutocorLen()`

* `volumetype::Symbol = :HyperRectangle`

* `max_startingIDs::Int = 10000`

* `max_startingIDs_fraction::Float64 = 2.5`

* `rect_increase::Float64 = 0.1`

* `warning_minstartingids::Int = 16`

* `dotrimming::Bool = true`

* `uncertainty::Vector{Symbol} = [:cov]`: List of uncertainty estimation methods
  to use, first entry will be used for primary result. Valid values:
    * `:cov`: Integral uncertainty for integration regions is estimated based
      on covariance of integrals of subsets of samples in the regions

    * `:ess`: Integral uncertainty for integration regions is estimated based
      on estimated effective number of samples in each region.
"""
@with_kw struct AHMIntegration{
    WA<:WhiteningAlgorithm,
    AC<:AutocorLenAlgorithm
} <: IntegrationAlgorithm
    whitening::WA = CholeskyPartialWhitening()
    autocorlen::AC = GeyerAutocorLen()
    volumetype::Symbol = :HyperRectangle
    max_startingIDs::Int = 10000
    max_startingIDs_fraction::Float64 = 2.5
    rect_increase::Float64 = 0.1
    warning_minstartingids::Int = 16
    dotrimming::Bool = true
    uncertainty::Vector{Symbol} = [:cov]
end
export AHMIntegration


"""
    bat_integrate(
        posterior::BAT.AnyPosterior,
        algorithm::IntegrationAlgorithm = AHMIntegration()
    )::DensitySampleVector

Calculate the integral (evidence) of `posterior`.

Returns a NamedTuple: (result = x::Measurement.Measurement, ...)

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.

`posterior` may be a

* [`BAT.AbstractPosteriorDensity`](@ref)

* [`BAT.DistLikeDensity`](@ref)

* [`BAT.DensitySampleVector`](@ref)

* `Distributions.MultivariateDistribution`

Uses the AHMI algorithm by default.
"""
function bat_integrate end
export bat_integrate


bat_integrate(posterior::AnyPosterior) = bat_integrate(posterior, AHMIntegration())


function bat_integrate(posterior::DensitySampleVector, algorithm::AHMIntegration)
    hmi_data = HMIData(unshaped.(posterior))
    
    integrationvol = algorithm.volumetype

    uncertainty_est_mapping = Dict(
        :cov => ("cov_weighted" => hm_combineresults_covweighted!),
        :ess => ("ess_weighted" => hm_combineresults_analyticestimation!),
    )

    uncertainty_estimators = Dict(uncertainty_est_mapping[u] for u in algorithm.uncertainty)

    primary_uncertainty_estimator = uncertainty_est_mapping[first(algorithm.uncertainty)][1]

    hmi_settings = HMISettings(
        _amhi_whitening_func(algorithm.whitening),
        algorithm.max_startingIDs,
        algorithm.max_startingIDs_fraction,
        algorithm.rect_increase,
        true,
        algorithm.warning_minstartingids,
        algorithm.dotrimming,
        uncertainty_estimators
    )

    hm_integrate!(hmi_data, integrationvol, settings = hmi_settings)

    result = hmi_data.integralestimates[primary_uncertainty_estimator].final
    info = hmi_data.integralestimates

    integral = Measurements.measurement(result.estimate, result.uncertainty)
    (result = integral, info = info)
end


function bat_integrate(posterior::AnyPosterior, algorithm::AHMIntegration)
    npar = totalndof(varshape(posterior))
    nsamples = 10^5 * npar
    samples = bat_sample(posterior, nsamples).result::DensitySampleVector
    bat_integrate(samples, algorithm)
end
#Union{PosteriorDensity, MultivariateDistribution}
