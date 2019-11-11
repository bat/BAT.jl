# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Add AbstractIntegratorAlgorithm or similar

"""
    bat_integrate(
        posterior::BAT.AnyPosterior,
    )::DensitySampleVector

Calculate the integral (evidence) of `posterior`.

Returns a NamedTuple: (integral = x::Measurement.Measurement, ...)

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


function bat_integrate(posterior::DensitySampleVector)
    hmi_data = HMIData(posterior)
    hm_integrate!(hmi_data)
    Z_signal_v = hmi_data.integralestimates["cov. weighted result"].final.estimate
    result = hmi_data.integralestimates["cov. weighted result"].final
    info = hmi_data.integralestimates

    integral = Measurements.measurement(result.estimate, result.uncertainty)
    (integral = integral, info = info)
end


function bat_integrate(posterior::AnyPosterior)
    npar = totalndof(params_shape(posterior))
    nsamples = 10^5 * npar
    samples = bat_sample(posterior, nsamples).samples::DensitySampleVector
    bat_integrate(samples)
end
