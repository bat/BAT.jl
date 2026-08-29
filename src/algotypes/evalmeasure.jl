# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    BAT.evalmeasure_impl(
        em::EvaluatedMeasure,
        algorithm,
        context::BATContext
    )::EvaluatedMeasure

*Experimental feature, not part of stable public API.*

Used internally by [`evalmeasure`](@ref). Specialize `BAT.evalmeasure_impl`
to implement new measure/distribution evaluation algorithms.

Implementations receive the evaluation target as an
[`EvaluatedMeasure`](@ref) and return an updated evaluated measure for the
same underlying measure, so `unevaluated(result) === unevaluated(em)`.
The result is constructed via `EvaluatedMeasure(em; ...)`; implementations
decide themselves which entries of `em` to overwrite and which to only
fill if absent, and they must record their evaluation by setting
`evalinfo = MeasureEvalInfo(algorithm, ...)`. Implementations that produce
transformed-space content pass their `transform_intent` and `f_transform`
in the same update, and report sample pairs stamped with the hash of that
transformation (see [`BAT.BispacedMeasure`](@ref)).
"""
function evalmeasure_impl end


"""
    evalmeasure(
        target::Union{AbstractMeasure,Distribution,DensitySampleVector},
        [algorithm],
        [context::BATContext]
    )::EvaluatedMeasure

Evaluate measure or probability distribution `target` using `algorithm` and
return an [`EvaluatedMeasure`](@ref).

If no algorithm is given, a default will be chosen depending on the type
of `target`. Typically, this will be an algorithm that draws (correlated
or uncorrelated) samples from `target`, and may also yield an approximation
of `target` and other estimates.

# Implementation

`evalmeasure` internally runs [`evalmeasure_impl`](@ref). Do not specialize
`evalmeasure` directly, specialize `evalmeasure_impl` instead to implement
new algorithms.
"""
function evalmeasure end
export evalmeasure

function convert_for(::typeof(evalmeasure), target)
    try
        convert(EvaluatedMeasure, batmeasure(target))
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("Can't convert target of type $(nameof(typeof(target))) to a BAT-compatible measure for `evalmeasure`: $(sprint(showerror, err))"))
    end
end

function evalmeasure(target, algorithm0, context::BATContext)
    em = convert_for(evalmeasure, target)::EvaluatedMeasure
    algorithm = batalgorithm(algorithm0)
    new_em = evalmeasure_impl(em, algorithm, context)::EvaluatedMeasure
    unevaluated(new_em) === unevaluated(em) || throw(ArgumentError("evalmeasure_impl for algorithm $(nameof(typeof(algorithm))) returned an EvaluatedMeasure of a different measure"))
    return new_em
end

function evalmeasure(target::MeasureLike)
    evalmeasure(target, get_batcontext())
end

function evalmeasure(target::MeasureLike, algorithm)
    evalmeasure(target, algorithm, get_batcontext())
end

function evalmeasure(target::MeasureLike, context::BATContext)
    em = convert_for(evalmeasure, target)
    algorithm = bat_default_withinfo(evalmeasure, Val(:algorithm), em)
    evalmeasure(em, algorithm, context)
end


function argchoice_msg(::typeof(evalmeasure), ::Val{:algorithm}, x)
    "Using measure evaluation algorithm $x"
end
