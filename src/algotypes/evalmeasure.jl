# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    evalmeasure(
        target::Union{AbstractMeasure,Distribution}
        [algorithm::BAT.AbstractSamplingAlgorithm],
        [context::BATContext]
    )::EvaluatedMeasure

Evaluate measure or probability distribution `target` using `algorithm` and
return an [`EvaluatedMeasure`](@ref).

If no algorithm is given, default will be chosen depending on the the type
of `target`. Typically, this will be an algorithm that draws (correlated
or uncorrelated) samples from `target`, and may also yield an approximation
of `target` and other estimates.


# Implementation

`evalmeasure` internally runs [`evalmeasure_impl`](@ref), then post-processes
the result via []`evalmeasure_postproc`](@ref).

Do not specialize `evalmeasure` directly but specalize `evalmeasure_impl`
instead to implement new algorithms.
"""
function evalmeasure end
export evalmeasure

function convert_for(::typeof(evalmeasure), target)
    try
        batmeasure(target)
    catch
        throw(ArgumentError("Can't convert $operation target of type $(nameof(typeof(target))) to a BAT-compatible measure for `evalmeasure`."))
    end
end


"""
    BAT.evalmeasure_impl(
        target::BATMeasure
        algorithm::BAT.AbstractSamplingAlgorithm,
        context::BATContext
    )::EvaluatedMeasure

Used internally by [`evalmeasure`](@ref). Specialize `BAT.evalmeasure.impl`
to implement new measure/distribution evaluation algorithms.
"""
function evalmeasure_impl end


"""
    BAT.evalmeasure_postproc(
        target::BATMeasure
        algorithm::BAT.AbstractSamplingAlgorithm,
        context::BATContext
    )::EvaluatedMeasure

Used internally by [`evalmeasure`](@ref).
"""
function evalmeasure_impl end


function evalmeasure(target, algorithm::AbstractSamplingAlgorithm, context::BATContext)
    measure = convert_for(evalmeasure, target)
    #!!!!!!!!!!! Append eval history in evalmeasure, or leave this to evalmeasure_impl?
    # orig_context = deepcopy(context)
    em_tmp = evalmeasure_impl(measure, algorithm, context)
    em = revalmeasure_postproc(measure, em_tmp)
    return em
end

function evalmeasure(target::AnySampleable)
    measure = convert_for(evalmeasure, target)
    evalmeasure(measure, get_batcontext())
end

function evalmeasure(target::AnySampleable, algorithm::AbstractSamplingAlgorithm)
    evalmeasure(target, algorithm, get_batcontext())
end

function evalmeasure(target::AnySampleable, context::BATContext)
    measure = convert_for(evalmeasure, target)
    algorithm = bat_default_withinfo(evalmeasure, Val(:algorithm), measure)
    evalmeasure(target, algorithm, context)
end


function argchoice_msg(::typeof(evalmeasure), ::Val{:algorithm}, x::AbstractSamplingAlgorithm)
    "Using evalmeasure algorithm $x"
end
