# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type BAT.AbstractSamplingAlgorithm

Abstract type for BAT sampling algorithms. See [`bat_sample`](@ref).
"""
abstract type AbstractSamplingAlgorithm end


"""
    bat_sample(
        target::BAT.MeasureLike,
        [algorithm::BAT.AbstractSamplingAlgorithm],
        [context::BATContext]
    )

Draw samples from `target` using `algorithm`.

Depending on sampling algorithm, the samples may be independent or correlated
(e.g. when using MCMC).

Returns a NamedTuple of the shape

```julia
(result = X::DensitySampleVector,)
```

Use [`evalmeasure`](@ref) instead to obtain an [`EvaluatedMeasure`](@ref)
that carries the samples together with all other evaluation results.

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
        throw(ArgumentError("Can't convert target of type $(nameof(typeof(target))) to a BAT-compatible measure for `bat_sample`: $(sprint(showerror, err))"))
    end
end


function bat_sample(target, algorithm::AbstractSamplingAlgorithm, context::BATContext)
    orig_context = deepcopy(context)

    em = evalmeasure(target, algorithm, context)
    smpls = samplesof(em)
    isnothing(smpls) && throw(ErrorException("Sampling algorithm $(nameof(typeof(algorithm))) did not produce samples"))
    r = (;result = smpls)

    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

function bat_sample(target::MeasureLike)
    measure = convert_for(bat_sample, target)
    bat_sample(measure, get_batcontext())
end

function bat_sample(target::MeasureLike, algorithm::AbstractSamplingAlgorithm)

    bat_sample(target, algorithm, get_batcontext())
end

function bat_sample(target::MeasureLike, context::BATContext)
    algorithm = bat_default_withinfo(bat_sample, Val(:algorithm), target)
    bat_sample(target, algorithm, context)
end


function argchoice_msg(::typeof(bat_sample), ::Val{:algorithm}, x::AbstractSamplingAlgorithm)
    "Using sampling algorithm $x"
end
