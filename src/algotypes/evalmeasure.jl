# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct EvalMeasureImplReturn

*Experimental feature, not part of stable public API yet.*

Return type of [`BAT.evalmeasure_impl`](@ref).

Constructor:

```julia
EvalMeasureImplReturn(;
    empirical::Union{DensitySampleMeasure,Nothing,Unchanged} = unchanged,
    approx::Union{BATMeasure,Nothing,Unchanged} = unchanged,
    dof::Union{IntegerLike,Nothing,Unchanged} = unchanged,
    mass::Union{Real,MeasureBase.AbstractUnknownMass,Unchanged} = unchanged,
    modes::Union{AbstractVector,Nothing,Unchanged} = unchanged,
    samplegen::Union{AbstractSampleGenerator,Nothing,Unchanged} = unchanged,
    evalresult::NamedTuple = (;),
)
```

Algorithms report the knowledge they produced in the corresponding fields.
`unchanged` (the default) leaves the current content of the evaluated
measure untouched, `nothing` (resp. `MeasureBase.UnknownMass()` for `mass`)
explicitly invalidates it.

The contents of EvalMeasureImplReturn is not part of the stable public API
and subject to change without notice.
"""
struct EvalMeasureImplReturn{
    S<:Union{DensitySampleMeasure,Nothing,Unchanged},
    A<:Union{BATMeasure,Nothing,Unchanged},
    N<:Union{IntegerLike,Nothing,Unchanged},
    U<:Union{Real,MeasureBase.AbstractUnknownMass,Unchanged},
    P<:Union{AbstractVector,Nothing,Unchanged},
    G<:Union{AbstractSampleGenerator,Nothing,Unchanged},
    R<:NamedTuple,
}
    _empirical::S
    _approx::A
    _dof::N
    _mass::U
    _modes::P
    _samplegen::G
    _evalresult::R
end

function EvalMeasureImplReturn(;
    empirical::Union{DensitySampleMeasure,Nothing,Unchanged} = unchanged,
    approx::Union{BATMeasure,Nothing,Unchanged} = unchanged,
    dof::Union{IntegerLike,Nothing,Unchanged} = unchanged,
    mass::Union{Real,MeasureBase.AbstractUnknownMass,Unchanged} = unchanged,
    modes::Union{AbstractVector,Nothing,Unchanged} = unchanged,
    samplegen::Union{AbstractSampleGenerator,Nothing,Unchanged} = unchanged,
    evalresult::NamedTuple = (;),
)
    return EvalMeasureImplReturn(empirical, approx, dof, mass, modes, samplegen, evalresult)
end


"""
    BAT.evalmeasure_impl(
        m::BATMeasure,
        algorithm,
        context::BATContext
    )::Union{EvalMeasureImplReturn,BATMeasure}

*Experimental feature, not part of stable public API yet.*

Used internally by [`evalmeasure`](@ref). Specialize `BAT.evalmeasure_impl`
to implement new measure/distribution evaluation algorithms.
"""
function evalmeasure_impl end

"""
    BAT.evalmeasure_postproc(
        orig_m::BATMeasure,
        result::Union{EvalMeasureImplReturn,BATMeasure},
        algorithm,
        orig_context::BATContext
    )::EvaluatedMeasure

*Experimental feature, not part of stable public API yet.*

Used internally by [`evalmeasure`](@ref).
"""
function evalmeasure_postproc end

function evalmeasure_postproc(
    orig_m::BATMeasure,
    eval_return::EvalMeasureImplReturn,
    algorithm,
    ::BATContext
)
    return EvaluatedMeasure(
        orig_m;
        empirical = eval_return._empirical,
        approx = eval_return._approx,
        dof = eval_return._dof,
        mass = eval_return._mass,
        modes = eval_return._modes,
        samplegen = eval_return._samplegen,
        evalinfo = MeasureEvalInfo(algorithm, eval_return._evalresult)
    )
end

function evalmeasure_postproc(::BATMeasure, eval_return::BATMeasure, ::Any, ::BATContext)
    return convert(EvaluatedMeasure, eval_return)
end


"""
    evalmeasure(
        target::Union{AbstractMeasure,Distribution}
        [algorithm],
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
the result via [`evalmeasure_postproc`](@ref).

Do not specialize `evalmeasure` directly but specialize `evalmeasure_impl`
instead to implement new algorithms.
"""
function evalmeasure end
export evalmeasure

function convert_for(::typeof(evalmeasure), target)
    try
        batmeasure(target)
    catch
        throw(ArgumentError("Can't convert target of type $(nameof(typeof(target))) to a BAT-compatible measure for `evalmeasure`."))
    end
end

function evalmeasure(target, algorithm0, context::BATContext)
    orig_m = convert_for(evalmeasure, target)::BATMeasure
    algorithm = batalgorithm(algorithm0)
    orig_context = deepcopy(context)
    eval_return = evalmeasure_impl(orig_m, algorithm, context)
    new_em = evalmeasure_postproc(orig_m, eval_return, algorithm, orig_context)
    return new_em
end

function evalmeasure(target::MeasureLike)
    measure = convert_for(evalmeasure, target)
    evalmeasure(measure, get_batcontext())
end

function evalmeasure(target::MeasureLike, algorithm)
    evalmeasure(target, algorithm, get_batcontext())
end

function evalmeasure(target::MeasureLike, context::BATContext)
    measure = convert_for(evalmeasure, target)
    algorithm = bat_default_withinfo(evalmeasure, Val(:algorithm), measure)
    evalmeasure(target, algorithm, context)
end


function argchoice_msg(::typeof(evalmeasure), ::Val{:algorithm}, x)
    "Using measure evaluation algorithm $x"
end
