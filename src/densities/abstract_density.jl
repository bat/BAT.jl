# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type BATDensity <: Function

*BAT-internal, not part of stable public API.*
"""
abstract type BATDensity <: Function end
@inline DensityInterface.DensityKind(::BATDensity) = IsDensity()

BATDensity(density::BATDensity) = density
BATDensity(density::Any) = WrappedNonBATDensity(density)

BATDensity(l::MeasureBase.Likelihood) = BATDensity(logfuncdensity(Base.Fix2(logdensityof, l.x) ∘ l.k))

Base.convert(::Type{BATDensity}, density::BATDensity) = density
Base.convert(::Type{BATDensity}, density::Any) = BATDensity(density)

Base.:∘(density::BATDensity, f::Function) = logdensityof(density) ∘ f


"""
    struct BAT.WrappedNonBATDensity

*BAT-internal, not part of stable public API.*
"""
struct WrappedNonBATDensity{D} <: BATDensity
    _d::D

    function WrappedNonBATDensity{D}(density::D) where D
        @argcheck DensityKind(density) isa IsDensity
        new{D}(density)
    end

end

WrappedNonBATDensity(density) = _wrapdensity_impl(density, DensityKind(density))

_wrapdensity_impl(density::D, ::IsDensity) where D = WrappedNonBATDensity(D)(density)
_wrapdensity_impl(density::Function, ::NoDensity) = WrappedNonBATDensity(logfuncdensity(logvalof ∘ density))
_wrapdensity_impl(density, ::NoDensity) = throw(ArgumentError("Can't wrap an object of type $(nameof(typeof(density))) as a density."))
_wrapdensity_impl(density, ::HasDensity) where D = throw(ArgumentError("Can't wrap an measure-like object of type $(nameof(typeof(density))) as a density."))

@inline Base.parent(density::WrappedNonBATDensity) = density._d

@inline DensityInterface.logdensityof(density::WrappedNonBATDensity, x) = logdensityof(parent(density), x)
@inline DensityInterface.logdensityof(density::WrappedNonBATDensity) = logdensityof(parent(density))

function Base.show(io::IO, density::WrappedNonBATDensity)
    print(io, Base.typename(typeof(density)).name, "(")
    show(io, parent(density))
    print(io, ")")
end



"""
    logdensityof_batmeasure(obj, x)

*BAT-internal, not part of stable public API.*
"""
function logdensityof_batmeasure end


"""
    abstract type BATMeasure <: AbstractMeasure

*BAT-internal, not part of stable public API.*
"""
abstract type BATMeasure <: AbstractMeasure end
@inline DensityInterface.DensityKind(::BATMeasure) = HasDensity()

BATMeasure(measure::BATMeasure) = measure
BATMeasure(dist::Distribution) = DistMeasure(dist)
BATMeasure(density::MeasureBase.DensityMeasure) = PosteriorMeasure(density.f, density.base)

Base.convert(::Type{BATMeasure}, measure::BATMeasure) = measure
Base.convert(::Type{BATMeasure}, measure) = BATMeasure(measure)

DensityInterface.logdensityof(m::BATMeasure, x) = logdensityof_batmeasure(m, x)
MeasureBase.logdensity_def(m::BATMeasure, x) = logdensityof_batmeasure(m, x)
MeasureBase.unsafe_logdensityof(m::BATMeasure, x) = logdensityof_batmeasure(m, x)

MeasureBase.basemeasure(m::BATMeasure) = _varshape_basemeasure(varshape(m))

_varshape_basemeasure(vs::ArrayShape{<:Real,1}) = MeasureBase.LebesgueBase()^length(vs)

@static if isdefined(MeasureBase, :NoFastInsupport)
    MeasureBase.insupport(m::BATMeasure, ::Any) = MeasureBase.NoFastInsupport{typeof(m)}()
else
    # Workaround:
    MeasureBase.insupport(m::BATMeasure, ::Any) = true
end

function ValueShapes.unshaped(measure::BATMeasure, vs::AbstractValueShape)
    varshape(measure) <= vs || throw(ArgumentError("Shape of density not compatible with given shape"))
    unshaped(measure)
end

show_value_shape(io::IO, vs::AbstractValueShape) = show(io, vs)
function show_value_shape(io::IO, vs::NamedTupleShape)
    print(io, Base.typename(typeof(vs)).name, "(")
    show(io, propertynames(vs))
    print(io, "}(…)")
end

function Base.show(io::IO, d::BATMeasure)
    print(io, Base.typename(typeof(d)).name, "(objectid = ")
    show(io, objectid(d))
    vs = varshape(d)
    if !ismissing(vs)
        print(io, ", varshape = ")
        show_value_shape(io, vs)
    end
    print(io, ")")
end



"""
    abstract type AnyBATDensityOrMeasure = Union{BATMeasure, BATDensity}

*BAT-internal, not part of stable public API.*
"""
const AnyBATDensityOrMeasure = Union{BATMeasure, BATDensity}


"""
    struct EvalException <: Exception

Constructors:

* ```$(FUNCTIONNAME)(func::Function, density::AnyBATDensityOrMeasure, v::Any, ret::Any)```

Fields:

$(TYPEDFIELDS)
"""
struct EvalException{F<:Function,D<:AnyBATDensityOrMeasure,V,C} <: Exception
    "Density evaluation function that failed."
    func::F

    "Density being evaluated."
    density::D

    "Variate at which the evaluation of `density` (applying `f` to `d` at `v`) failed."
    v::V

    "Cause of failure, either the invalid return value of `f` on `d` at `v`, or another expection (on rethrow)."
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
    print(io, ", density is ")
    show(io, err.density)
end


#!!!!!!!!!!!!! strip var_bounds out
#var_bounds

#!!!!!!!!!!!!!!! rip out
#dist_param_bounds


# ToDo: Handle this better:

function ValueShapes.totalndof(measure::BATMeasure)
    shape = varshape(measure)
    ismissing(shape) ? missing : ValueShapes.totalndof(shape)
end

ValueShapes.varshape(::BATMeasure) = missing

# ToDo: Handle this better:
bat_sampler(d::AnyMeasureLike) = Distributions.sampler(d)


"""
    checked_logdensityof(measure::BATMeasure, v::Any, T::Type{<:Real})

*BAT-internal, not part of stable public API.*

Evaluates density log-value via `DensityInterface.logdensityof` and performs
additional checks.

Throws a `BAT.EvalException` on any of these conditions:

* The variate shape of `measure` (if known) does not match the shape of `v`.
* The return value of `DensityInterface.logdensityof` is `NaN`.
* The return value of `DensityInterface.logdensityof` is an equivalent of positive
  infinity.
"""
function checked_logdensityof end

function checked_logdensityof(density::BATMeasure, v::Any)
    logval = try
        logdensityof(density, v)
    catch err
        @rethrow_logged EvalException(logdensityof, density, v, err)
    end

    _check_density_logval(density, v, logval)


    return logval
end

ZygoteRules.@adjoint checked_logdensityof(density::BATMeasure, v::Any) = begin
    check_variate(varshape(density), v)
    logval, back = try
        ZygoteRules.pullback(logdensityof(density), v)
    catch err
        @rethrow_logged EvalException(logdensityof, density, v, err)
    end
    _check_density_logval(density, v, logval)
    eval_logval_pullback(logval::Real) = (nothing, first(back(logval)))
    (logval, eval_logval_pullback)
end

function _check_density_logval(density::BATMeasure, v::Any, logval::Real)
    if isnan(logval) || !(logval < float(typeof(logval))(+Inf))
        @throw_logged(EvalException(logdensityof, density, v, logval))
    end
    nothing
end

function ChainRulesCore.rrule(::typeof(_check_density_logval), density::BATMeasure, v::Any, logval::Real)
    return _check_density_logval(density, v, logval), _check_density_logval_pullback
end
_check_density_logval_pullback(ΔΩ) = (NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent())



value_for_msg(v::Real) = v
# Strip dual numbers to make errors more readable:
value_for_msg(v::ForwardDiff.Dual) = ForwardDiff.value(v)
value_for_msg(v::AbstractArray) = value_for_msg.(v)
value_for_msg(v::NamedTuple) = map(value_for_msg, v)


"""
    BAT.density_valtype(density::BATMeasure, v::Any)

*BAT-internal, not part of stable public API.*

Determine a suitable return type for the (log-)density value
of the given density for the given variate.
"""
function density_valtype end

@inline function density_valtype(density::BATMeasure, v::Any)
    T = float(realnumtype(typeof((v))))
    promote_type(T, default_val_numtype(density))
end

function ChainRulesCore.rrule(::typeof(density_valtype), density::BATMeasure, v::Any)
    result = density_valtype(density, v)
    _density_valtype_pullback(ΔΩ) = (NoTangent(), NoTangent(), ZeroTangent())
    return result, _density_valtype_pullback
end


"""
    BAT.default_var_numtype(density::BATMeasure)

*BAT-internal, not part of stable public API.*

Returns the default/preferred underlying numerical type for (elements of)
variates of `density`.
"""
function default_var_numtype end
default_var_numtype(density::BATMeasure) = Float64


"""
    BAT.default_val_numtype(density::BATMeasure)

*BAT-internal, not part of stable public API.*

Returns the default/preferred numerical type (log-)density values of
`density`.
"""
function default_val_numtype end
default_val_numtype(density::BATMeasure) = Float64


"""
    BAT.log_zero_density(T::Type{<:Real})

log-density value to assume for regions of implicit zero density, e.g.
outside of variate/parameter bounds/support.

Returns an equivalent of negative infinity.
"""
log_zero_density(T::Type{<:Real}) = float(T)(-Inf)


"""
    BAT.is_log_zero(x::Real, T::Type{<:Real} = typeof(x)}

*BAT-internal, not part of stable public API.*

Check if x is an equivalent of log of zero, resp. negative infinity,
in respect to type `T`.
"""
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



"""
    abstract type DistLikeMeasure <: BATMeasure

*BAT-internal, not part of stable public API.*

A density that implements part of the `Distributions.Distribution` interface.
Such densities are suitable for use as a priors.

Typically, custom priors should be implemented as subtypes of
`Distributions.Distribution`. BAT will automatically wrap them in a subtype of
`DistLikeMeasure`.

Subtypes of `DistLikeMeasure` are required to support more functionality
than an [`BATMeasure`](@ref), but less than a
`Distribution{Multivariate,Continuous}`.

A `d::Distribution{Multivariate,Continuous}` can be converted into (wrapped
in) an `DistLikeMeasure` via `conv(DistLikeMeasure, d)`.

The following functions must be implemented for subtypes:

* `BAT.logdensityof_batmeasure`

* `ValueShapes.varshape`

* `Distributions.sampler`

* `Statistics.cov`
"""
abstract type DistLikeMeasure <: BATMeasure end




"""
    BAT.MeasureLike = Union{...}

Union of all types that BAT will accept as a probability measure
(though not all algorithms will work with all types):
    
* `MeasureBase.AbstractMeasure`
* `Distributions.ContinuousDistribution`
"""
const AnyMeasureLike = Union{
    MeasureBase.AbstractMeasure,
    Distributions.ContinuousDistribution
}
export AnyMeasureLike


"""
    BAT.AnySampleable = Union{...}

Union of all types that BAT can sample from:

* [`AnyMeasureLike`](@ref)
* [`DensitySampleVector`](@ref)
* `Distributions.Distribution`
"""
const AnySampleable = Union{
    AnyMeasureLike,
    Distributions.Distribution,
    DensitySampleVector
}
export AnySampleable


"""
    BAT.AnyIIDSampleable = Union{...}

*BAT-internal, not part of stable public API.*

Union of all distribution/density-like types that BAT can draw i.i.d.
(independent and identically distributed) samples from:

* [`DistLikeMeasure`](@ref)
* `Distributions.Distribution`
"""
const AnyIIDSampleable = Union{
    DistMeasure,
    Distributions.Distribution,
}
