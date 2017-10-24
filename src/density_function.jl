# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using FunctionWrappers: FunctionWrapper


# ToDo: Add `density_logval_gradient!` to support HMC, etc.



doc"""
    AbstractDensityFunction

The following functions must be implemented for subtypes:

* `BAT.nparams`
* `BAT.density_logval`

In some cases, it may be desirable to override the default implementations
of the functions

* `BAT.exec_capabilities`
* `BAT.density_logval!`

By default, an `AbstractDensityFunction` is unboundeded. Parameter bounds can
be set by overriding the default implementation of `BAT.param_bounds` or by
wrapping the density in a `BoundedDensity`.
"""
abstract type AbstractDensityFunction end
export AbstractDensityFunction


param_bounds(density::AbstractDensityFunction) = UnboundedParams{Float64}()


doc"""
    density_logval(
        density::AbstractDensityFunction,
        params::AbstractVector{<:Real},
        exec_context::ExecContext = ExecContext()
    )

See `ExecContext` for thread-safety requirements.
"""
function density_logval end
export density_logval

# Assume that density_logval isn't always thread-safe, but usually remote-safe:
exec_capabilities(::typeof(density_logval), density::AbstractDensityFunction, params::AbstractVector{<:Real}) =
    ExecCapabilities(0, false, 0, true)


doc"""
    density_logval!(
        r::AbstractArray{<:Real},
        density::AbstractDensityFunction,
        params::AbstractMatrix{<:Real},
        exec_context::ExecContext = ExecContext()
    )

Compute log of values of a density function for multiple parameter value
vectors.

Input:

* `density`: density function
* `params`: parameter values (column vectors)
* `exec_context`: Execution context

Output is stored in

* `r`: Array of log-result values, length must match, shape is ignored

Array size requirements:

    size(params, 1) == length(r)

The result of `density_logval!` for parameter vectors that are out of bounds
is undefined: `density_logval!` must not fail, but may store arbitrary values
in the corresponding entries of `r`.

See `ExecContext` for thread-safety requirements.
"""
function density_logval! end
export density_logval!


function density_logval!(
    r::AbstractArray{<:Real},
    density::AbstractDensityFunction,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext = ExecContext()
)
    # TODO: Support for parallel execution
    single_ec = exec_context # Simplistic, will have to change for parallel execution
    for i in eachindex(r, indices(params, 2))
        p = view(params, :, i) # TODO: Avoid memory allocation
        r[i] = density_logval(density, p, single_ec)
    end
    r
end

# ToDo: Derive from exec_capabilities(density_logval, density, ...)
exec_capabilities(::typeof(density_logval!), density::AbstractDensityFunction, params::AbstractMatrix{<:Real}) =
    ExecCapabilities(0, false, 0, true) # Change when default implementation of density_logval! for AbstractDensityFunction becomes multithreaded.



struct GenericDensityFunction{F} <: AbstractDensityFunction
    log_f::F
    nparams::Int
end

export GenericDensityFunction

Base.parent(density::GenericDensityFunction) = density.log_f

nparams(density::GenericDensityFunction) = density.nparams

function density_logval(
    density::GenericDensityFunction,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    density.log_f(params)
end



#=

mutable struct TransformedDensity{
    SO<:AbstractDensityFunction,
    SN<:AbstractDensityFunction
} <: AbstractDensityFunction
   before::SO
   after::SN
   # ... transformation, Jacobi matrix of transformation, etc.
end

export TransformedDensity

...

=#
