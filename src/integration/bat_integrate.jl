# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Add AbstractIntegratorAlgorithm or similar

"""
    bat_integrate(
        posterior::BAT.AnyPosterior,
    )::PosteriorSampleVector

Calculate the integral (evidence) of `posterior`.

`posterior` may be a

* [`BAT.AbstractPosteriorDensity`](@ref)

* [`BAT.DistLikeDensity`](@ref)

* [`BAT.PosteriorSampleVector`](@ref)

* `Distributions.MultivariateDistribution`

Uses the AHMI algorithm by default.
"""
function bat_integrate end
export bat_integrate


function bat_integrate(posterior::PosteriorSampleVector)
    hmi_data = HMIData(posterior)
    hm_integrate!(hmi_data)
    Z_signal_v = hmi_data.integralestimates["cov. weighted result"].final.estimate
    result = hmi_data.integralestimates["cov. weighted result"].final
    Measurements.measurement(result.estimate, result.uncertainty)
end


function bat_integrate(posterior::AnyPosterior)
    npar = totalndof(params_shape(posterior))
    nsamples = 10^5 * npar
    samples = bat_sample(posterior, nsamples)::PosteriorSampleVector
    bat_integrate(samples)
end
