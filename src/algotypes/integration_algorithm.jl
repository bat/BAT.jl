# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type IntegrationAlgorithm

Abstract type for integration algorithms.
"""
abstract type IntegrationAlgorithm end
export IntegrationAlgorithm


"""
    bat_integrate(
        target::MeasureLike,
        [algorithm::IntegrationAlgorithm],
        [context::BATContext]
    )

Calculate the integral (evidence) of `target`.

Returns a NamedTuple of the shape

```julia
(result = X,)
```

where `X` is the mass estimate, typically a `Measurements.Measurement` or
a logarithmic number type wrapping one (e.g. for nested-sampling evidence
estimates).

Use [`evalmeasure`](@ref) instead to obtain an [`EvaluatedMeasure`](@ref)
that carries the mass estimate together with all other evaluation results.

# Implementation

`bat_integrate` uses [`evalmeasure`](@ref) internally. Do not specialize
`bat_integrate`.
"""
function bat_integrate end
export bat_integrate


function bat_integrate(target::MeasureLike, algorithm::IntegrationAlgorithm, context::BATContext)
    orig_context = deepcopy(context)
    em = evalmeasure(target, algorithm, context)
    mass = massof(em)
    mass isa MeasureBase.AbstractUnknownMass && throw(ErrorException("Integration algorithm $(nameof(typeof(algorithm))) did not produce a mass estimate"))
    r = (;result = mass)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_integrate(target::MeasureLike) = bat_integrate(target, get_batcontext())

function bat_integrate(target::MeasureLike, algorithm::IntegrationAlgorithm)
    bat_integrate(target, algorithm, get_batcontext())
end

function bat_integrate(target::MeasureLike, context::BATContext)
    algorithm::IntegrationAlgorithm = bat_default_withinfo(bat_integrate, Val(:algorithm), target)
    bat_integrate(target, algorithm, context)
end


function argchoice_msg(::typeof(bat_integrate), ::Val{:algorithm}, x::IntegrationAlgorithm)
    "Using integration algorithm $x"
end
