# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Distributions
using FunctionWrappers
import FunctionWrappers: FunctionWrapper


"""
    AbstractTargetDensity

The following functions must be implemented for subtypes:

* `BAT.target_logval`

In some cases, it may be desirable to override the default implementations
of the functions

* `BAT.exec_capabilities`
* `BAT.target_logval!`
"""
abstract type AbstractTargetDensity end
export AbstractTargetDensity

# Optional target_(re-)init(target, exec_context)??


"""
    target_logval!(
        r::AbstractArray{<:Real},
        target::AbstractTargetDensity,
        params::AbstractMatrix{<:Real},
        exec_context::ExecContext = ExecContext()
    )

Compute log of values of target density for multiple parameter value vectors.

Input:

* `params`: parameter values (column vectors)
* `bounds`: Parameter bounds
* `exec_context`: Execution context

Output is stored in

* `r`: Array of log-result values, length must match, shape is ignored

Array size requirements:

    size(params, 1) == length(r)

See `ExecContext` for thread-safety requirements.
"""
function target_logval! end
export target_logval!


function target_logval!(
    r::AbstractArray{<:Real},
    target::AbstractTargetDensity,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext = ExecContext()
)
    # TODO: Support for parallel execution
    single_ec = exec_context # Simplistic, will have to change for parallel execution
    for i in eachindex(r, indices(params, 2))
        p = view(params, :, i) # TODO: Avoid memory allocation
        r[i] = target_logval(target, p, single_ec)
    end
    r
end

# Assume that target_logval isn't always thread-safe, but usually remote-safe:
exec_capabilities(::typeof(target_logval!), target::AbstractTargetDensity, params::AbstractMatrix{<:Real}) =
    ExecCapabilities(0, false, 0, true) # Change when default implementation of target_logval! for AbstractTargetDensity becomes multithreaded.


"""
    target_logval(
        target::AbstractTargetDensity,
        params::AbstractVector{<:Real},
        exec_context::ExecContext = ExecContext()
    )

See `ExecContext` for thread-safety requirements.
"""
function target_logval end
export target_logval


# Assume that target_logval isn't always thread-safe, but usually remote-safe:
exec_capabilities(::typeof(target_logval), target::AbstractTargetDensity, params::AbstractVector{<:Real}) =
    ExecCapabilities(0, false, 0, true)



struct GenericTargetDensity{F} <: AbstractTargetDensity
    log_f::F
end

export GenericTargetDensity

function target_logval(
    target::GenericTargetDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    target.log_f(params)
end
