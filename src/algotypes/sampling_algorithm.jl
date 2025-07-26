# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type BAT.AbstractSamplingAlgorithm

Abstract type for BAT sampling algorithms. See [`bat_sample`](@ref).
"""
abstract type AbstractSamplingAlgorithm end


"""
    bat_sample(
        target::BAT.AnySampleable,
        [algorithm::BAT.AbstractSamplingAlgorithm],
        [context::BATContext]
    )::DensitySampleVector

Draw samples from `target` using `algorithm`.

Depending on sampling algorithm, the samples may be independent or correlated
(e.g. when using MCMC).

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector, evaluated::EvaluatedMeasure, ...)
```
(The field `evaluated` is only present if `target` is a measure.)

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

# Implementation

`bat_sample` uses [`evalmeasure`](@ref) internally. Do not specialize
`bat_sample`.
"""
function bat_sample end
export bat_sample


function convert_for(::typeof(bat_sample), target)
    try
        batmeasure(target)
    catch err
        throw(ArgumentError("Can't convert $operation target of type $(nameof(typeof(target))) to a BAT-compatible measure."))
    end
end


function bat_sample(target, algorithm::AbstractSamplingAlgorithm, context::BATContext)
    orig_context = deepcopy(context)

    em = evalmeasure(target, algorithm, context)
    r = (;result = samplesof(em), evalinfo(em).result...)

    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

function bat_sample(target::AnySampleable)
    measure = convert_for(bat_sample, target)
    bat_sample(measure, get_batcontext())
end

function bat_sample(target::AnySampleable, algorithm::AbstractSamplingAlgorithm)

    bat_sample(target, algorithm, get_batcontext())
end

function bat_sample(target::AnySampleable, context::BATContext)
    algorithm = bat_default_withinfo(bat_sample, Val(:algorithm), target)
    bat_sample(target, algorithm, context)
end


function argchoice_msg(::typeof(bat_sample), ::Val{:algorithm}, x::AbstractSamplingAlgorithm)
    "Using sampling algorithm $x"
end
