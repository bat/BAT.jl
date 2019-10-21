# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Add `density_logvalgrad!` to support HMC, etc.


@doc """
    AbstractDensity

Subtypes of `AbstractDensity` only have to imlement the function

* `BAT.density_logval`

However, densities with known parameters bounds should also implement

* `BAT.param_bounds`

If the parameter bounds are unkown, but the number of parameters is known,
the function

* `BAT.nparams`

should be implemented directly (usually it is inferred from the bounds).

Densities that support named parameters should also implement

* `BAT.params_shape`
"""
abstract type AbstractDensity end
export AbstractDensity


@doc """
    density_logval(density::AbstractDensity, params::AbstractVector{<:Real})
    density_logval(density::AbstractDensity, params::NamedTuple)

Compute log of value of a multi-variate density function at the given
parameter values.

Input:

* `density`: density function
* `params`: parameter values

Subtypes of `AbstractDensity` must implement `density_logval` for either
`params::AbstractVector{<:Real}` or `params::NamedTuple`.

Note: If `density_logval` is called with out-of-bounds parameters (see
`param_bounds`), the behaviour is undefined. The result for parameters that
are not within bounds is *implicitly* `-Inf`, but it is the caller's
responsibility to handle these cases.
"""
function density_logval end
export density_logval


@doc """
    param_bounds(
        density::AbstractDensity
    )::Union{AbstractParamBounds,Missing}

Get the parameter bounds of `density`. See `density_logval` for the
implications and handling of the bounds. If the bounds are missing,
`density_logval` must be prepared to handle any parameter values.
"""
param_bounds(density::AbstractDensity) = missing
export param_bounds


@doc """
    nparams(density::AbstractDensity)::Union{Int,Missing}

Get the number of parameters of `density`. May return `missing`, if the
density supports a variable number of parameters.
"""
function nparams(density::AbstractDensity)
    bounds = param_bounds(density)
    ismissing(bounds) ? missing : nparams(bounds)
end


@doc """
    params_shape(
        density::AbstractDensity
    )::Union{ValueShapes.AbstractValueShape,Missing}       #!!! FORMER ::Union{ValueShapes.AbstractValueShape,Missing,Nothing}

    params_shape(
        density::AbstractPriorDensity
    )::ValueShapes.AbstractValueShape      #!!! FORMER ::Union{ValueShapes,Nothing}

Get the shapes of parameters of `density`.

For prior densities, the result must not be `missing`, but may be `nothing` if
the prior only supports flat parameter vectors.
"""
function params_shape end
export params_shape

params_shape(density::AbstractDensity) = missing


@doc """
    eval_density_logval(
        density::AbstractDensity,
        params::AbstractVector{<:Real},
        parshapes::ValueShapes.AbstractValueShape   #!!! FORMER Union{ValueShapes.AbstractValueShape,Nothing}
    )

Internal function to evaluate density log-value, calls `density_logval`.

`parshapes` *must* be compatible with `params_shape(density)`.

Checks that:

* The number of parameters of `density` (if known) matches the length of
  `params`.
* The return value of `density_logval` is not `NaN`.
* The return value of `density_logval` is less than `+Inf`.
"""
function eval_density_logval(
    density::AbstractDensity,
    params::AbstractVector{<:Real},
    parshapes::ValueShapes.AbstractValueShape              #! FORMER Union{ValueShapes.AbstractValueShape,Nothing}
)
    npars = nparams(density)
    ismissing(npars) || (length(eachindex(params)) == npars) || throw(ArgumentError("Invalid length of parameter vector"))

    r = float(density_logval(density, _apply_parshapes(params, parshapes)))
    isnan(r) && throw(ErrorException("Return value of density_logval must not be NaN, density has type $(typeof(density))"))
    r < convert(typeof(r), +Inf) || throw(ErrorException("Return value of density_logval must not be posivite infinite, density has type $(typeof(density))"))

    r
end

# !!!! FORMER _apply_parshapes(params::AbstractVector{<:Real}, parshapes::Nothing) = params

_apply_parshapes(params::AbstractVector{<:Real}, parshapes::AbstractValueShape) = parshapes(params)


@doc """
    AbstractPriorDensity <: AbstractDensity

A density suitable for use as a prior.

Subtypes of `AbstractPriorDensity` are required to support more functionality
than a `AbstractDensity`, but less than a
`Distribution{Multivariate,Continuous}`.

The following functions must be implemented for subtypes:

* `BAT.density_logval`

* `BAT.param_bounds`

* `BAT.params_shape`

* `Distributions.sampler`

* `Statistics.cov`

Prior densities that support named parameters should also implement

* `BAT.params_shape`

A `d::Distribution{Multivariate,Continuous}` can be converted into (wrapped
in) an `AbstractPriorDensity` via `conv(AbstractPriorDensity, d)`.
"""
abstract type AbstractPriorDensity <: AbstractDensity end
export AbstractPriorDensity


@doc """
    param_bounds(density::AbstractPriorDensity)::AbstractParamBounds

Get the parameter bounds of `density`. Must not be `missing`.
"""
function param_bounds end


@doc """
    nparams(density::AbstractPriorDensity)::Int

Get the number of parameters of prior density `density`. Must not be
`missing`, prior densities must have a fixed number of parameters. By default,
the number of parameters is inferred from the parameter bounds.
"""
nparams(density::AbstractPriorDensity) = nparams(param_bounds(density))
