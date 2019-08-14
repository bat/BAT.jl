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

* `BAT.param_shapes`
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

function density_logval(density::AbstractDensity, params::AbstractVector{<:Real})
    ps = param_shapes(density)
    density_logval(density, ps(params))
end


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
    if ismissing(bounds)
        missing
    else
        nparams(bounds)
    end
end


@doc """
    param_shapes(
        density::AbstractDensity
    )::Union{ShapesOfVariables.VarShapes,Missing}

Get the shapes of parameters of `density`. Must return a
`ShapesOfVariables.VarShapes` object.
"""
function param_shapes(density::AbstractDensity)
    missing
end
export param_shapes



@doc """
    eval_density_logval!(
        T::Type{<:Real},
        density::AbstractDensity,
        params::AbstractVector{<:Real},
    )

Apply bounds and then evaluate density and check return value.

May modify `params` to force them into bounds.

Guarantees that for out-of-bounds parameters:

* `density_logval` is not called
* log value of density is set to (resp. returned as) `-Inf`
"""
function eval_density_logval! end

function eval_density_logval!(
    T::Type{<:Real},
    density::AbstractDensity,
    params::AbstractVector{<:Real};
    do_applybounds::Bool = true
)
    npars = nparams(density)
    ismissing(npars) || (length(eachindex(params)) == npars) || throw(ArgumentError("Invalid length of parameter vector"))

    bounds = param_bounds(density)
    if !ismissing(bounds) && do_applybounds
        apply_bounds!(params, bounds)
    end
    if ismissing(bounds) || !isoob(params)
        ismissing(bounds) || @assert params in bounds  # TODO: Remove later on for increased performance, should never trigger
        r = density_logval(density, params)
        isnan(r) && throw(ErrorException("Return value of density_logval must not be NaN"))  
        T(r)
    else
        T(-Inf)
    end
end



@doc """
    AbstractPriorDensity <: AbstractDensity

A density suitable for use as a prior.

Subtypes of `AbstractPriorDensity` are required to support more functionality
than a `AbstractDensity`, but less than a
`Distribution{Multivariate,Continuous}`.

The following functions must be implemented for subtypes:

* `BAT.density_logval`

* `BAT.param_bounds`

* `BAT.param_shapes`

* `Distributions.sampler`

* `Statistics.cov`

Prior densities that support named parameters should also implement

* `BAT.param_shapes`

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


@doc """
    param_shapes(density::AbstractPriorDensity)::ShapesOfVariables

Get the shapes of parameters of `density`. Must not be `missing`.
"""
function param_shapes end


# ToDo: Implement rand and rand! for prior to override rand(rng::AbstractRNG, X)