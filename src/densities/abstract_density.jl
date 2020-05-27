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
    eval_density_logval(
        density::AbstractDensity,
        v::AbstractVector{<:Real},
        shape::ValueShapes.AbstractValueShape
    )

*BAT-internal, not part of stable public API.*

Evaluates density log-value via `density_logval`.

`shape` *must* be compatible with `ValueShapes.varshape(density)`.

Checks that:

* The variate shape of `density` (if known) matches shape of `v`.
* The return value of `density_logval` is not `NaN`.
* The return value of `density_logval` is less than `+Inf`.
"""
function eval_density_logval(
    density::AbstractDensity,
    v::AbstractVector{<:Real},
    shape::ValueShapes.AbstractValueShape
)
    npars = totalndof(density)
    ismissing(npars) || (length(eachindex(v)) == npars) || throw(ArgumentError("Invalid length of parameter vector"))

    v_shaped = _apply_parshapes(v, shape)
    raw_r = try
        density_logval(density, _apply_parshapes(v, shape))
    catch err
        v_real = _strip_duals(v)
        v_real_shaped = _apply_parshapes(v_real, shape)
        throw(ErrorException("Density evaluation failed at v = $v_real_shaped due to exception $err"))
    end
    r = float(raw_r)
    isnan(r) && throw(ErrorException("Return value of density_logval must not be NaN, density has type $(typeof(density))"))
    r < convert(typeof(r), +Inf) || throw(ErrorException("Return value of density_logval must not be posivite infinite, density has type $(typeof(density))"))

    r
end

_apply_parshapes(v::AbstractVector{<:Real}, shape::AbstractValueShape) = stripscalar(shape(v))

_strip_duals(x) = x
_strip_duals(x::AbstractVector{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)


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
