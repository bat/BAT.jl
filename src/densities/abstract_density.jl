# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractDensity

Subtypes of `AbstractDensity` must implement the function

* `BAT.eval_logval_unchecked`

For likelihood densities this is typically sufficient, since shape, and
variate bounds will be inferred from the prior.

Densities with a known variate shape may also implement

* `ValueShapes.varshape`

Densities with known variate bounds may also implement

* `BAT.var_bounds`

!!! note

    The function `BAT.var_bounds` is not part of the stable public BAT-API,
    it's name and arguments may change without deprecation.
"""
abstract type AbstractDensity end
export AbstractDensity


show_value_shape(io::IO, vs::AbstractValueShape) = show(io, vs)
function show_value_shape(io::IO, vs::NamedTupleShape)
    print(io, Base.typename(typeof(vs)).name, "(")
    show(io, propertynames(vs))
    print(io, "}(…)")
end

function Base.show(io::IO, d::AbstractDensity)
    print(io, Base.typename(typeof(d)).name, "(objectid = ")
    show(io, objectid(d))
    print(io, ", varshape = ")
    show_value_shape(io, varshape(d))
    print(io, ")")
end


"""
    struct DensityEvalException <: Exception

Constructors:

* ```$(FUNCTIONNAME)(func::Function, density::AbstractDensity, v::Any, ret::Any)```

Fields:

$(TYPEDFIELDS)
"""
struct DensityEvalException{F<:Function,D<:AbstractDensity,V,C} <: Exception
    "Density evaluation function that failed."
    func::F

    "Density being evaluated."
    density::D

    "Variate at which the evaluation of `density` (applying `f` to `d` at `v`) failed."
    v::V

    "Cause of failure, either the invalid return value of `f` on `d` at `v`, or another expection (on rethrow)."
    ret::C
end

function Base.showerror(io::IO, err::DensityEvalException)
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


"""
    BAT.eval_logval_unchecked(density::AbstractDensity, v::Any)

Compute log of the value of a multivariate density function for the given
variate/parameter-values.

Input:

* `density`: density function
* `v`: argument, i.e. variate / parameter-values

Note: If `eval_logval_unchecked` is called with an argument that is out of bounds,
the behavior is undefined. The result for arguments that are not within
bounds is *implicitly* `-Inf`, but it is the caller's responsibility to handle
these cases.
"""
function eval_logval_unchecked end


"""
    var_bounds(
        density::AbstractDensity
    )::Union{AbstractVarBounds,Missing}

*BAT-internal, not part of stable public API.*

Get the parameter bounds of `density`. See `eval_logval_unchecked` and
`eval_logval` for the implications and handling of bounds.
If bounds are missing, `eval_logval_unchecked` must be prepared to
handle any parameter values.
"""
var_bounds(density::AbstractDensity) = missing


"""
    ValueShapes.totalndof(density::AbstractDensity)::Union{Int,Missing}

Get the number of degrees of freedom of the variates of `density`. May return
`missing`, if the shape of the variates is not fixed.
"""
function ValueShapes.totalndof(density::AbstractDensity)
    shape = varshape(density)
    ismissing(shape) ? missing : ValueShapes.totalndof(shape)
end


"""
    ValueShapes.varshape(
        density::AbstractDensity
    )::Union{ValueShapes.AbstractValueShape,Missing}

    ValueShapes.varshape(
        density::DistLikeDensity
    )::ValueShapes.AbstractValueShape

Get the shapes of the variates of `density`.

For prior densities, the result must not be `missing`, but may be `nothing` if
the prior only supports unshaped variate/parameter vectors.
"""
ValueShapes.varshape(density::AbstractDensity) = missing


"""
    eval_logval(density::AbstractDensity, v::Any, T::Type{<:Real})

*BAT-internal, not part of stable public API.*

Evaluates density log-value via `eval_logval_unchecked`.

Throws a `BAT.DensityEvalException` on any of these conditions:

* The variate shape of `density` (if known) does not match the shape of `v`.
* The return value of `eval_logval_unchecked` is `NaN`.
* The return value of `eval_logval_unchecked` is an equivalent of positive
  infinity.
"""
function eval_logval end

@inline function eval_logval(density::AbstractDensity, v::Any, T::Type{<:Real})
    v_shaped = fixup_variate(varshape(density), v)
    R = density_logval_type(v_shaped, T)
    v_stripped = stripscalar(v_shaped)
    _generic_eval_logval_impl(density, v_stripped, R)
end


function _generic_eval_logval_impl(density::AbstractDensity, v::Any, T::Type)
    logval = try
        eval_logval_unchecked(density, v)
    catch err
        rethrow(DensityEvalException(eval_logval, density, v, err))
    end

    _check_density_logval(density, v, logval)

    return convert(T, logval)::T
end


function _check_density_logval(density::AbstractDensity, v::Any, logval::Real)
    if isnan(logval) || !(logval < float(typeof(logval))(+Inf))
        throw(DensityEvalException(eval_logval, density, v, logval))
    end

    nothing
end


value_for_msg(v::Real) = v
# Strip dual numbers to make errors more readable:
value_for_msg(v::ForwardDiff.Dual) = ForwardDiff.value(v)
value_for_msg(v::AbstractArray) = value_for_msg.(v)
value_for_msg(v::NamedTuple) = map(value_for_msg, v)


"""
    BAT.density_logval_type(v::Any, T::Type{<:Real})

*BAT-internal, not part of stable public API.*

Determine a suitable return type of log-density functions, given a variate
shape `vs` and a default result type `T`.
"""
function density_logval_type end

@inline function density_logval_type(v::Any, T::Type{<:Real})
    U = float(getnumtype(v))
    promote_type(T, U)
end

default_dlt() = Float32


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


# ToDO: Consider deprecation in favor of logdensityof
"""
    logvalof(density::AbstractDensity)::Function

Returns a function that computes the logarithmic value of `density` at a
given point:

```julia
f = logvalof(density)
log_density_at_v = f(v)
```
"""
logvalof(density::AbstractDensity) = logdensityof(density)


"""
    logdensityof(density::AbstractDensity, v)::Real
    logdensityof(density::AbstractDensity)::Function

*Experimental feature, not part of stable public API.*

Computes the logarithmic value of `density` at a given point, resp. returns a
function that does so:

```julia
logy = logdensityof(density, v)
logdensityof(density, v) == logdensityof(density)(v)
```

Note: This function should *not* be specialized for custom density types!
"""
function logdensityof end
export logdensityof

logdensityof(density::AbstractDensity, v::Any) = eval_logval(density, v, default_dlt())

logdensityof(density::AbstractDensity) = LogDensityOf(density)


struct LogDensityOf{D<:AbstractDensity} <: Function
    density::D
end

@inline (lvd::LogDensityOf)(v::Any) = logdensityof(lvd.density, v)

(Base.:-)(lvd::LogDensityOf) = NegLogDensityOf(lvd.density)

ValueShapes.varshape(lvd::LogDensityOf) = varshape(lvd.density)
ValueShapes.unshaped(lvd::LogDensityOf) = LogDensityOf(unshaped(lvd.density))


struct NegLogDensityOf{D<:AbstractDensity} <: Function
    density::D
end

@inline (lvd::NegLogDensityOf)(v::Any) = - logdensityof(lvd.density, v)

(Base.:-)(lvd::NegLogDensityOf) = LogDensityOf(lvd.density)

ValueShapes.varshape(lvd::NegLogDensityOf) = varshape(lvd.density)
ValueShapes.unshaped(lvd::NegLogDensityOf) = NegLogDensityOf(unshaped(lvd.density))


function Base.show(io::IO, f::Union{LogDensityOf,NegLogDensityOf})
    print(io, Base.typename(typeof(f)).name, "(")
    show(io, f.density)
    print(io, ")")
end

Base.show(io::IO, M::MIME"text/plain", f::Union{LogDensityOf,NegLogDensityOf}) = show(io, f)



"""
    abstract type DistLikeDensity <: AbstractDensity

A density that implements part of the `Distributions.Distribution` interface.
Such densities are suitable for use as a priors.

Typically, custom priors should be implemented as subtypes of
`Distributions.Distribution`. BAT will automatically wrap them in a subtype of
`DistLikeDensity`.

Subtypes of `DistLikeDensity` are required to support more functionality
than an [`AbstractDensity`](@ref), but less than a
`Distribution{Multivariate,Continuous}`.

A `d::Distribution{Multivariate,Continuous}` can be converted into (wrapped
in) an `DistLikeDensity` via `conv(DistLikeDensity, d)`.

The following functions must be implemented for subtypes:

* `BAT.eval_logval_unchecked`

* `ValueShapes.varshape`

* `Distributions.sampler`

* `Statistics.cov`

* `BAT.var_bounds`

!!! note

    The function `BAT.var_bounds` is not part of the stable public BAT-API
    and subject to change without deprecation.
"""
abstract type DistLikeDensity <: AbstractDensity end
export DistLikeDensity


"""
    var_bounds(density::DistLikeDensity)::AbstractVarBounds

*BAT-internal, not part of stable public API.*

Get the parameter bounds of `density`. Must not be `missing`.
"""
function var_bounds end


"""
    ValueShapes.totalndof(density::DistLikeDensity)::Int

Get the number of degrees of freedom of the variates of `density`. Must not be
`missing`, a `DistLikeDensity` must have a fixed variate shape.
"""
ValueShapes.totalndof(density::DistLikeDensity) = totalndof(var_bounds(density))



"""
    BAT.AnyDensityLike = Union{...}

Union of all types that BAT will accept as a probability density, resp. that
`convert(AbstractDensity, d)` supports:
    
* [`AbstractDensity`](@ref)
* `Distributions.Distribution`
"""
const AnyDensityLike = Union{
    AbstractDensity,
    Distributions.ContinuousDistribution
}
export AnyDensityLike


"""
    BAT.AnySampleable = Union{...}

Union of all types that BAT can sample from:

* [`AbstractDensity`](@ref)
* [`DensitySampleVector`](@ref)
* `Distributions.Distribution`
"""
const AnySampleable = Union{
    AbstractDensity,
    DensitySampleVector,
    Distributions.Distribution
}
export AnySampleable


"""
    BAT.AnyIIDSampleable = Union{...}

Union of all distribution/density-like types that BAT can draw i.i.d.
(independent and identically distributed) samples from:

* [`DistLikeDensity`](@ref)
* `Distributions.Distribution`
"""
const AnyIIDSampleable = Union{
    DistLikeDensity,
    Distributions.Distribution
}
export AnyIIDSampleable
