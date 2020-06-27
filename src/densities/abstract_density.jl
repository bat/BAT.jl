# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    AbstractDensity

Subtypes of `AbstractDensity` must implement the function

* `BAT.density_logval`

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
    BAT.density_logval(density::AbstractDensity, v::Any)

Compute log of the value of a multivariate density function for the given
variate/parameter-values.

Input:

* `density`: density function
* `v`: argument, i.e. variate / parameter-values

Note: If `density_logval` is called with an argument that is out of bounds,
the behaviour is undefined. The result for arguments that are not within
bounds is *implicitly* `-Inf`, but it is the caller's responsibility to handle
these cases.
"""
function density_logval end


@doc doc"""
    BAT.density_logvalgrad(density::AbstractDensity, v::AbstractVector{<:Real})

Compute the log of the value of a multivariate density function, as well as
the gradient of the log-value for the given variate/parameter-values.

Input:

* `density`: density function
* `v`: argument, i.e. variate / parameter-values

Returns a tuple of the log density value and it's gradient.

See also [`BAT.density_logval`](@ref).
"""
function density_logvalgrad(
    density::AbstractDensity,
    v::AbstractVector{<:Real},
)
    n = length(eachindex(v))
    log_f = v -> BAT.density_logval(density, v)

    P = eltype(v)
    T = typeof(log_f(v)) # Inefficient

    grad_logd = Vector{P}(undef, n)
    chunk = ForwardDiff.Chunk(v)
    config = ForwardDiff.GradientConfig(log_f, v, chunk)
    result = DiffResults.MutableDiffResult(zero(T), (grad_logd,))
    ForwardDiff.gradient!(result, log_f, v, config)
    logd = DiffResults.value(result)
    (logd = logd, grad_logd = grad_logd)
end


@doc doc"""
    var_bounds(
        density::AbstractDensity
    )::Union{AbstractVarBounds,Missing}

*BAT-internal, not part of stable public API.*

Get the parameter bounds of `density`. See `density_logval` for the
implications and handling of the bounds. If the bounds are missing,
`density_logval` must be prepared to handle any parameter values.
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
    eval_density_logval!(
        density::AbstractDensity,
        v::Any,
        T::Type{:Real} = density_logval_type(v);
        use_bounds::Bool = true,
        apply_bounds::Bool = false,
        strict::Bool = false
    )::T

*BAT-internal, not part of stable public API.*

Evaluates density log-value via `density_logval`.

May modify `v` with `apply_bounds = true`, will leave `v` unmodified
otherwise.

Throws an exception on any of these conditions:

* The variate shape of `density` (if known) does not match the shape of `v`.
* The return value of `density_logval` is `NaN`.
* The return value of `density_logval` is an equivalent of positive
  infinity.

Options:

* `use_bounds`: Use the bounds of `density` to explicitly return an
  equivalent of negative infinity if `v` is out of bounds.

* `apply_bounds`: Try to modify `v` to make `v` fall within the bounds of
  `density`, by applying transformations inherit in the type of the bounds
  (e.g. for reflective or cyclic bounds).

* `strict`: Throw an exception if `v` is out of bounds (after applying
  transformations if `apply_bounds = true`).
"""
function eval_density_logval!(
    density::AbstractDensity,
    v::Any,
    T::Type{<:Real} = density_logval_type(v);
    use_bounds::Bool = true,
    apply_bounds::Bool = false,
    strict::Bool = false
)
    v_shaped = get_shaped_variate(density, v)
    if ! process_shaped_variate!(density, v, T, use_bounds, apply_bounds, strict)
        return log_zero_density(T)
    end

    logval::T = try
        density_logval(density, stripscalar(v_shaped))
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
        throw(ErrorException("Return value of density_logval must not be NaN, v = $(variate_for_msg(v)) , density has type $(typeof(density))"))
    end

    if !(logval < typeof(logval)(+Inf))
        throw(ErrorException("Return value of density_logval must not be posivite infinite, v = $(variate_for_msg(v)), density has type $(typeof(density))"))
    end

    nothing
end


function get_shaped_variate(density::AbstractDensity, v::Any)
    shape = valshape(v)
    expected_shape = varshape(density)
    if !ismissing(expected_shape) && shape != expected_shape
        throw(ArgumentError("Shape of variate doesn't match variate shape of density, with variate of type $(typeof(v)) and density of type $(typeof(density))"))
    end
    v
end

function get_shaped_variate(density::AbstractDensity, v::AbstractVector{<:Real})
    shape = varshape(density)
    ismissing(shape) ? v : shape(v)
end


function process_shaped_variate!(density::AbstractDensity, v_shaped::Any, T::Type{<:Real}, use_bounds::Bool, apply_bounds::Bool, strict::Bool)
    shape = valshape(v_shaped)

    v_unshaped = unshaped(v_shaped)
    if unshaped(v_shaped) !== v_unshaped
        throw(ArgumentError("Shaped variate of type $(typeof(v_shaped)) does not support unshaped views"))
    end

    npars = length(eachindex(v_unshaped))
    npars_expected = totalndof(density)
    if !ismissing(npars_expected) && npars != npars_expected
        throw(ArgumentError("Invalid length ($npars) of parameter vector, density has $npars_expected degrees of freedom and type $(typeof(density))"))
    end

    bounds = var_bounds(density)
    if !ismissing(bounds)
        if apply_bounds
            apply_bounds!(v_unshaped, bounds)
        end

        if !(v_unshaped in bounds)
            if strict
                throw(ArgumentError("Parameter(s) out of bounds, density has type $(typeof(density))"))
            else
                if use_bounds
                    return false
                end
            end
        end
    end

    return true
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
    BAT.density_logval_type(v::Any, T::Type{<:Real} = Float32)

*BAT-internal, not part of stable public API.*

Determine a suitable return type of log-density functions, given a variate
`v` and an optional additional default result type `T`.
"""
function density_logval_type end

@inline function density_logval_type(v::AbstractArray{<:Real}, T::Type{<:Real} = Float32)
    vs = valshape(v)
    U = float(eltype(v))
    promote_type(T, U)
end

@inline function density_logval_type(v::Any, T::Type{<:Real} = Float32)
    vs = valshape(v)
    U = float(ValueShapes.default_unshaped_eltype(vs))
    promote_type(T, U)
end


@doc doc"""
    BAT.log_zero_density(T::Type{<:Real})

log-density value to assume for regions of implicit zero density, e.g.
outside of variate/parameter bounds/support.

Returns an equivalent of negative infinity.
"""
log_zero_density(T::Type{<:Real}) = float(T)(-Inf)


@doc doc"""
    BAT.is_log_zero_density(x::Real, T::Type{<:Real} = typeof(x)}

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

* `BAT.density_logval`

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
