# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type BATDensity end
@inline DensityInterface.DensityKind(::BATDensity) = IsDensity()


_get_model(likelihood::Likelihood) = likelihood.k
_get_observation(likelihood::Likelihood) = likelihood.x

const _SimpleLikelihood = DensityInterface.LogFuncDensity{<:ComposedFunction{<:Base.Fix2{typeof(logdensityof),<:Any},<:Any}}
_get_model(likelihood::_SimpleLikelihood) = likelihood._log_f.inner
_get_observation(likelihood::_SimpleLikelihood) = likelihood._log_f.outer.x

_precompose_density(likelihood::Likelihood, g) = likelihoodof(ffcomp(_get_model(likelihood), g), _get_observation(likelihood))

function _precompose_density(density, g)
    @argcheck DensityKind(density) isa IsDensity
    logfuncdensity(ffcomp(logdensityof(density), g))
end


"""
    struct EvalException <: Exception

*Experimental feature, not part of stable public API.*

Constructors:

* ```$(FUNCTIONNAME)(func::Function, measure::Any, v::Any, ret::Any)```

The target `measure` may be a measure or a generic density. The fields retain
the supplied objects without copying them.

Fields:

$(TYPEDFIELDS)
"""
struct EvalException{F<:Function,D,V,C} <: Exception
    "Density evaluation function that failed."
    func::F

    "Measure or density being evaluated; the field retains its historical name."
    measure::D

    "Variate at which `func(measure, v)` failed."
    v::V

    "Cause of failure: the invalid return value of `func(measure, v)`, or the caught exception."
    ret::C
end

function Base.showerror(io::IO, err::EvalException)
    print(io, "Density evaluation with $(err.func) failed at ")
    show(io, value_for_msg(err.v))
    if err.ret isa Exception
        print(io, " due to exception ")
        showerror(io, err.ret)
    else
        print(io, ", must not evaluate to ")
        show(io, value_for_msg(err.ret))
    end
    print(io, ", measure is ")
    show(io, err.measure)
end


"""
    checked_logdensityof(target, v)
    checked_logdensityof(target)

*BAT-internal, not part of stable public API.*

Evaluates a measure or density via `DensityInterface.logdensityof` and performs
additional checks. The one-argument form returns a function of the variate.

The generic method preserves valid `Real` results, including negative infinity,
without conversion. It throws a `BAT.EvalException` on any of these conditions:

* `DensityInterface.logdensityof` throws an exception.
* The return value of `DensityInterface.logdensityof` is `NaN`.
* The return value of `DensityInterface.logdensityof` is an equivalent of positive
  infinity.

The Zygote rule checks variate shape before evaluation; errors from that check
are not wrapped. Errors raised later by the reverse pullback are also not
wrapped. Specialized measure methods may impose different checks.
"""
function checked_logdensityof end

@inline checked_logdensityof(target) = Base.Fix1(checked_logdensityof, target)

@inline DensityInterface.logfuncdensity(f::Base.Fix1{typeof(checked_logdensityof)}) = f.x

function checked_logdensityof(target, v)
    logval = try
        logdensityof(target, v)
    catch err
        @rethrow_logged EvalException(logdensityof, target, v, err)
    end

    _check_density_logval(target, v, logval)

    #R = density_valtype(measure, v_shaped)
    #return convert(R, logval)::R
    return logval
end

ZygoteRules.@adjoint checked_logdensityof(target, v) = begin
    check_variate(varshape(target), v)
    logval, back = try
        ZygoteRules.pullback(logdensityof(target), v)
    catch err
        @rethrow_logged EvalException(logdensityof, target, v, err)
    end
    _check_density_logval(target, v, logval)
    function eval_logval_pullback(logval::Real)
        tangents = back(logval)
        tangent = isnothing(tangents) ? nothing : first(tangents)
        (nothing, isnothing(tangent) && v isa AbstractArray{<:Real} ? zero(v) : tangent)
    end
    (logval, eval_logval_pullback)
end

function _check_density_logval(target, v, logval::Real)
    if isnan(logval) || !(logval < float(typeof(logval))(+Inf))
        @throw_logged(EvalException(logdensityof, target, v, logval))
    end
    nothing
end

function ChainRulesCore.rrule(::typeof(_check_density_logval), target, v::Any, logval::Real)
    return _check_density_logval(target, v, logval), _check_density_logval_pullback
end
_check_density_logval_pullback(::Any) = (NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent())



value_for_msg(v) = v
# Strip dual numbers to make errors more readable:
value_for_msg(v::ForwardDiff.Dual) = ForwardDiff.value(v)
value_for_msg(v::AbstractArray) = value_for_msg.(v)
value_for_msg(v::NamedTuple) = map(value_for_msg, v)


"""
    BAT.log_zero_density(T::Type{<:Real})

*BAT-internal, not part of stable public API.*

log-density value to assume for regions of implicit zero density, e.g.
outside of variate/parameter bounds/support.

Returns an equivalent of negative infinity.
"""
log_zero_density(T::Type{<:Real}) = float(T)(-Inf)


"""
    BAT.is_log_zero(x::Real, T::Type = typeof(x)}

*BAT-internal, not part of stable public API.*

Check if x is an equivalent of log of zero, resp. negative infinity,
in respect to type `T`.
"""
function is_log_zero end

function is_log_zero(x::Real, T::Type{<:Real} = typeof(x))
    U = typeof(x)

    FT = float(T)
    FU = float(U)

    x_notnan = !isnan(x)
    x_isinf = !isfinite(x)
    x_isneg = x < zero(x)
    x_notgt1 = !(x > log_zero_density(FT))
    x_notgt2 = !(x > log_zero_density(FU))
    x_iseq1 = x ≈ log_zero_density(FT)
    x_iseq2 = x ≈ log_zero_density(FU)

    x_notnan && ((x_isinf && x_isneg) | x_notgt1 | x_notgt2 | x_iseq1 | x_iseq1)
end

is_log_zero(x::Real, T::Type) = is_log_zero(x, typeof(x))


@inline function density_valtype(target::T, v::U) where {T,U}
    Core.Compiler.return_type(logdensityof, Tuple{T,U})
end

function ChainRulesCore.rrule(::typeof(density_valtype), target, v)
    result = density_valtype(target, v)
    _density_valtype_pullback(::Any) = (NoTangent(), NoTangent(), ZeroTangent())
    return result, _density_valtype_pullback
end


convert_density_value(::Type{T}, dval) where {T<:Real} = convert(T, dval)::T
convert_density_value(::Type, dval) = dval
