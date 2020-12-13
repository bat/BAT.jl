# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    IntegrationAlgorithm

Abstract type for integration algorithms.
"""
abstract type IntegrationAlgorithm end
export IntegrationAlgorithm


"""
    bat_integrate(
        target::AnySampleable,
        [algorithm::IntegrationAlgorithm]
    )::DensitySampleVector

Calculate the integral (evidence) of `target`.

Returns a NamedTuple of the shape

```julia
(result = X::Measurements.Measurement, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_integrate`, add methods to
    `bat_integrate_impl` instead.
"""
function bat_integrate end
export bat_integrate

function bat_integrate_impl end


function bat_integrate(
    target::AnySampleable,
    algorithm::IntegrationAlgorithm = bat_default_withinfo(bat_integrate, Val(:algorithm), target)
)
    r = bat_integrate_impl(target, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_integrate), ::Val{:algorithm}, x::IntegrationAlgorithm)
    "Using integration algorithm $x"
end
