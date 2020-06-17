# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    IntegrationAlgorithm

Abstract type for integration algorithms.
"""
abstract type IntegrationAlgorithm end
export IntegrationAlgorithm


"""
    bat_integrate(
        posterior::BAT.AnyPosterior,
        [algorithm::IntegrationAlgorithm]
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


@inline function bat_integrate(target::AnyPosterior)
    algorithm = bat_default_withinfo(bat_integrate, Val(:algorithm), target)
    r = bat_integrate(target, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_integrate), ::Val{:algorithm}, x::IntegrationAlgorithm)
    "Using integration algorithm $x"
end
