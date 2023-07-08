# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type IntegrationAlgorithm

Abstract type for integration algorithms.
"""
abstract type IntegrationAlgorithm end
export IntegrationAlgorithm


"""
    bat_integrate(
        target::AnySampleable,
        [algorithm::IntegrationAlgorithm],
        [context::BATContext]
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


function bat_integrate(target::AnySampleable, algorithm::IntegrationAlgorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_integrate_impl(target, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_integrate(target::AnySampleable) = bat_integrate(target, get_batcontext())

function bat_integrate(target::AnySampleable, algorithm::IntegrationAlgorithm)
    bat_integrate(target, algorithm, get_batcontext())
end

function bat_integrate(target::AnySampleable, context::BATContext)
    algorithm::IntegrationAlgorithm = bat_default_withinfo(bat_integrate, Val(:algorithm), target)
    bat_integrate(target, algorithm, context)
end


function argchoice_msg(::typeof(bat_integrate), ::Val{:algorithm}, x::IntegrationAlgorithm)
    "Using integration algorithm $x"
end
