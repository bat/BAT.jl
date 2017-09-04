# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Distributions
using FunctionWrappers
import FunctionWrappers: FunctionWrapper


# ToDo: Rename to ...TargetDensity?
"""
    AbstractTargetDensity

The following functions must be implemented for subtypes:

* `BAT.target_logval`

In some cases, it may be desirable to override the default implementations
of the functions

* `BAT.exec_compat`
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

    size(params,1) == length(r)
"""
function target_logval! end
export target_logval!


function target_logval!(
    r::AbstractArray{<:Real},
    target::AbstractTargetDensity,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext = ExecContext()
)
    # TODO: Parallel execution, depending on exec_context and exec_compat(target, target_logval, params)
    single_ec = exec_context # Simplistic, will have to change for parallel execution
    for i in eachindex(r, indices(params, 2))
        p = view(params, :, i) # TODO: Avoid memory allocation
        r[i] = target_logval(target, p, single_ec)
    end
    r
end


"""
    target_logval(
        target::AbstractTargetDensity,
        params::AbstractVector{<:Real},
        exec_context::ExecContext = ExecContext()
    )

The caller must not assume that `target_logval` is thread-safe.
"""
function target_logval end
export target_logval


"""
    exec_compat(func::Function, target::AbstractTargetDensity, args...)
"""
function exec_compat end
export exec_compat

exec_compat(::typeof(target_logval), target::AbstractTargetDensity, params::AbstractVector{<:Real}) =
    ExecCompat(false, 0)

exec_compat(::typeof(target_logval!), target::AbstractTargetDensity, params::AbstractMatrix{<:Real}) =
    ExecCompat(false, 0) # Will need to change when default implementation of target_logval! gets support for parallel execution



struct ConstTargetDensity{T<:Real} <: AbstractTargetDensity
    log_value::T
end

export ConstTargetDensity

function target_logval(
    target::ConstTargetDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    target.log_value
end


# function target_logval!(
#     r::AbstractArray{<:Real},
#     target::ConstTargetDensity,
#     params::AbstractMatrix{<:Real},
#     exec_context::ExecContext
# )
#     @assert size(params, 2) == length(r)
#     fill!(r, target.log_value)
# end





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



struct MvDistTargetDensity{D<:Distribution{Multivariate,Continuous}} <: AbstractTargetDensity
    d::D
end

export MvDistTargetDensity


function target_logval(
    target::MvDistTargetDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    Distributions.logpdf(target.d, params)
end


function target_logval!(
    r::AbstractArray{<:Real},
    target::MvDistTargetDensity,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext = ExecContext()
)
    # TODO: Parallel execution, depending on exec_context
    Distributions.logpdf!(r, target.d, params)
end


exec_compat(::typeof(target_logval!), target::MvDistTargetDensity, params::AbstractMatrix{<:Real}) =
    ExecCompat(false, 0) # Will need to change when implementation of target_logval! gets support for parallel execution



struct GenericProductTargetDensity{T<:Real,P<:Real}
    log_terms::Vector{FunctionWrapper{T,Tuple{P}}}
    single_exec_compat::ExecCompat
end

export GenericProductTargetDensity


function target_logval(
    target::GenericProductTargetDensity{T,P},
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
) where {T,P}
    # TODO: Use exec_context and target.exec_compat
    sum((log_term(convert(P, T)) for (log_term, p) in (target.log_terms, params)))
end


exec_compat(::typeof(target_logval), target::GenericProductTargetDensity, params::AbstractVector{<:Real}) =
    target.single_exec_compat


# ToDo: Add product of target densitys
