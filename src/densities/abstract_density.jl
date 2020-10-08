# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    AbstractDensity

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
    it's name and arguments may change without notice.
"""
abstract type AbstractDensity end
export AbstractDensity


@doc doc"""
    BAT.eval_logval_unchecked(density::AbstractDensity, v::Any)

Compute log of the value of a multivariate density function for the given
variate/parameter-values.

Input:

* `density`: density function
* `v`: argument, i.e. variate / parameter-values

Note: If `eval_logval_unchecked` is called with an argument that is out of bounds,
the behaviour is undefined. The result for arguments that are not within
bounds is *implicitly* `-Inf`, but it is the caller's responsibility to handle
these cases.
"""
function eval_logval_unchecked end


@doc doc"""
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


@doc doc"""
    BAT.estimate_finite_bounds(density::AbstractDensity)
    BAT.estimate_finite_bounds(dist::Distribution)

*BAT-internal, not part of stable public API.*

Estimate finite bounds of a density.
Currently, the bounds are estimated by calculating the 0.00001 and 0.99999 quantiles.
"""
function estimate_finite_bounds end


@doc doc"""
    ValueShapes.totalndof(density::AbstractDensity)::Union{Int,Missing}

Get the number of degrees of freedom of the variates of `density`. May return
`missing`, if the shape of the variates is not fixed.
"""
function ValueShapes.totalndof(density::AbstractDensity)
    bounds = var_bounds(density)
    ismissing(bounds) ? missing : ValueShapes.totalndof(bounds)
end


@doc doc"""
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


@doc doc"""
    eval_logval(
        density::AbstractDensity,
        v::Any,
        T::Type{:Real} = density_logval_type(v);
        use_bounds::Bool = true,
        strict::Bool = false
    )::T

*BAT-internal, not part of stable public API.*

Evaluates density log-value via `eval_logval_unchecked`.

Throws an exception on any of these conditions:

* The variate shape of `density` (if known) does not match the shape of `v`.
* The return value of `eval_logval_unchecked` is `NaN`.
* The return value of `eval_logval_unchecked` is an equivalent of positive
  infinity.

Options:

* `use_bounds`: Return an equivalent of negative infinity if `v` is out of
  bounds. `use_bounds` should only be set to `false` when it is known that
  the `v` is within bounds.

* `strict`: Throw an exception if `v` is out of bounds.
"""
function eval_logval end

function eval_logval(
    density::AbstractDensity,
    v::Any,
    T::Type{<:Real} = density_logval_type(v);
    use_bounds::Bool = true,
    strict::Bool = false
)
    v_shaped = reshape_variate(varshape(density), v)
    if use_bounds && !variate_is_inbounds(density, v_shaped, strict)
        return log_zero_density(T)
    end

    # ToDo: Make Zygote-compatible, by wrapping the following exception
    # augmentation mechanism in a function `get_density_logval_with_rethrow`
    # with a custom pullback:
    logval::T = try
        eval_logval_unchecked(density, stripscalar(v_shaped))
    catch err
        rethrow(_density_eval_error(density, v, err))
    end

    _check_density_logval(density, v, logval, strict)

    return logval
end

function _density_eval_error(density::AbstractDensity, v::Any, err::Any)
    ErrorException("Density evaluation failed at v = $(variate_for_msg(v)) due to exception $err, density has type $(typeof(density))")
end

function _check_density_logval(density::AbstractDensity, v::Any, logval::Real, strict::Bool)
    if isnan(logval)
        throw(ErrorException("Log-density must not evaluate to NaN, v = $(variate_for_msg(v)) , density has type $(typeof(density))"))
    end

    if !(logval < typeof(logval)(+Inf))
        throw(ErrorException("Log-density must not evaluate to posivite infinity, v = $(variate_for_msg(v)), density has type $(typeof(density))"))
    end

    nothing
end


function variate_is_inbounds(density::AbstractDensity, v::Any, strict::Bool)
    bounds = var_bounds(density)
    if !ismissing(bounds) && !(v in bounds)
        if strict
            throw(ArgumentError("Parameter(s) out of bounds, density has type $(typeof(density))"))
        else
            return false
        end
    else
        true
    end
end


_strip_duals(x) = x
_strip_duals(x::AbstractVector{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)

function variate_for_msg(v::Any)
    # Strip dual numbers to make errors more readable:
    shape = valshape(v)
    v_real_unshaped = _strip_duals(unshaped(v))
    v_real_shaped = shape(v_real_unshaped)
    stripscalar(v_real_shaped)
end


@doc doc"""
    renormalize_variate!(v_renorm::Any, density::AbstractDensity, v::Any)

*BAT-internal, not part of stable public API.*
"""
function renormalize_variate!(v_renorm::Any, density::AbstractDensity, v::Any)
    renormalize_variate!(v_renorm, var_bounds(density), v)
end




@doc doc"""
    eval_gradlogval(density::AbstractDensity, v::AbstractVector{<:Real})
    grad
*BAT-internal, not part of stable public API.*

Compute the log of the value of a multivariate density function, as well as
the gradient of the log-value for the given variate/parameter-values.

Input:

* `density`: density function
* `v`: argument, i.e. variate / parameter-values

Returns a tuple of the log density value and it's gradient.

Note: This function should *not* be specialized for custom density types!
"""
function eval_gradlogval end


function eval_gradlogval(
    density::AbstractDensity,
    v::Any,
)
    log_f = logvalof(density)

    shape = valshape(v)
    v_unshaped = unshaped(v)

    n = length(eachindex(v_unshaped))
    P = eltype(v_unshaped)
    T = density_logval_type(v_unshaped)

    grad_logd_unshaped = Vector{P}(undef, n)
    chunk = ForwardDiff.Chunk(v_unshaped)
    config = ForwardDiff.GradientConfig(log_f, v_unshaped, chunk)
    result = DiffResults.MutableDiffResult(zero(T), (grad_logd_unshaped,))
    ForwardDiff.gradient!(result, log_f, v_unshaped, config)
    logd = DiffResults.value(result)

    gradshape = replace_const_shapes(ValueShapes.const_zero_shape, shape)

    grad_logd = gradshape(grad_logd_unshaped)
    (logd = logd, grad_logd = grad_logd)
end


@doc doc"""
    BAT.density_logval_type(v::Any, T::Type{<:Real} = Float32)

*BAT-internal, not part of stable public API.*

Determine a suitable return type of log-density functions, given a variate
`v` and an optional additional default result type `T`.
"""
function density_logval_type end

@inline function density_logval_type(v::AbstractArray{<:Real}, T::Type{<:Real} = Float32)
    U = float(eltype(v))
    promote_type(T, U)
end

@inline density_logval_type(v::Any, T::Type{<:Real} = Float32) = density_logval_type(unshaped(v), T)


@doc doc"""
    BAT.log_zero_density(T::Type{<:Real})

log-density value to assume for regions of implicit zero density, e.g.
outside of variate/parameter bounds/support.

Returns an equivalent of negative infinity.
"""
log_zero_density(T::Type{<:Real}) = float(T)(-Inf)


@doc doc"""
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



struct LogValOfDensity{D<:AbstractDensity} <: Function
    density::D
end

(lvd::LogValOfDensity)(v::Any) = eval_logval(lvd.density, v)


"""
    logvalof(density::AbstractDensity)::Function

Returns a function that computes that logarithmic value of `density` at a
given point:

```julia
    f = logvalof(density)
    log_density_at_v = f(v)
```
"""
logvalof(density::AbstractDensity) = LogValOfDensity(density)



struct LogValGradOfDensity{D<:AbstractDensity} <: Function
    density::D
end

(lvdg::LogValGradOfDensity)(v::Any) = eval_gradlogval(lvdg.density, v)


"""
    eval_gradlogval(density::AbstractDensity)::Function

Returns a function that is equivalent to

```julia
    v -> eval_gradlogval(density, v)
```
"""
eval_gradlogval(density::AbstractDensity) = LogValGradOfDensity(density)



@doc doc"""
    DistLikeDensity <: AbstractDensity

A density that implements part of the `Distributions.Distribution` interface.
Such densities are suitable to be used as a priors.

Typically, custom priors should be implemented as subtypes of
`Distributions.Distribution`. BAT will automatically wrap them in a subtype of
`DistLikeDensity`.

Subtypes of `DistLikeDensity` are required to support more functionality
than a `AbstractDensity`, but less than a
`Distribution{Multivariate,Continuous}`.

A `d::Distribution{Multivariate,Continuous}` can be converted into (wrapped
in) an `DistLikeDensity` via `conv(DistLikeDensity, d)`.

The following functions must be implemented for subtypes:

* `BAT.eval_logval_unchecked`

* `BAT.var_bounds`

* `ValueShapes.varshape`

* `Distributions.sampler`

* `Statistics.cov`

!!! note

    The function `BAT.var_bounds` is not part of the stable public BAT-API,
    it's name and arguments may change without notice.
"""
abstract type DistLikeDensity <: AbstractDensity end
export DistLikeDensity


@doc doc"""
    var_bounds(density::DistLikeDensity)::AbstractVarBounds

*BAT-internal, not part of stable public API.*

Get the parameter bounds of `density`. Must not be `missing`.
"""
function var_bounds end


@doc doc"""
    ValueShapes.totalndof(density::DistLikeDensity)::Int

Get the number of degrees of freedom of the variates of `density`. Must not be
`missing`, a `DistLikeDensity` must have a fixed variate shape.
"""
ValueShapes.totalndof(density::DistLikeDensity) = totalndof(var_bounds(density))



"""
    BAT.AnyDensityLike = Union{...}

Union of all types that BAT will accept as a probability Density, resp. that
`convert(AbstractDenstiy, d)` supports:
    
* [`AbstractDensity`](@ref)
* `Distributions.Distribution`
* `StatsBase.Histogram`
"""
const AnyDensityLike = Union{
    AbstractDensity,
    Distributions.ContinuousDistribution,
    StatsBase.Histogram
}
export AnyDensityLike


"""
    BAT.AnySampleable = Union{...}

Union of all types that BAT can sample from:

* [`AbstractDensity`](@ref)
* [`DensitySampleVector`](@ref)
* `Distributions.Distribution`
* `StatsBase.Histogram`
"""
const AnySampleable = Union{
    AbstractDensity,
    DensitySampleVector,
    Distributions.Distribution,
    StatsBase.Histogram
}
export AnySampleable


"""
    BAT.AnyIIDSampleable = Union{...}

Union of all distribution/density-like types that BAT can draw i.i.d.
(independent and identically distributed) samples from:

* [`DistLikeDensity`](@ref)
* `Distributions.Distribution`
* `StatsBase.Histogram`
"""
const AnyIIDSampleable = Union{
    DistLikeDensity,
    Distributions.Distribution,
    StatsBase.Histogram
}
export AnyIIDSampleable
