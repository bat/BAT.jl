# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Distributions
using FunctionWrappers
import FunctionWrappers: FunctionWrapper


# ToDo: Rename to ...TargetDensity?
"""
    AbstractTargetFunction

The following functions must be implemented for subtypes:

* `BAT.target_logval`

In some cases, it may be desirable to override the default implementations
of the functions

* `BAT.exec_compat`
* `BAT.target_logval!`
"""
abstract type AbstractTargetFunction end
export AbstractTargetFunction

# Optional target_(re-)init(target, exec_context)??


"""
    target_logval!(
        r::AbstractArray{<:Real},
        target::AbstractTargetFunction,
        params::AbstractMatrix{<:Real},
        exec_context::ExecContext = ExecContext()
    )

Blah ...

end

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
    target::AbstractTargetFunction,
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
        target::AbstractTargetFunction,
        params::AbstractVector{<:Real},
        exec_context::ExecContext = ExecContext()
    )

The caller must not assume that `target_logval` is thread-safe.
"""
function target_logval end
export target_logval


"""
    exec_compat(func::Function, target::AbstractTargetFunction, args...)
"""
function exec_compat end
export exec_compat

exec_compat(::typeof(target_logval), target::AbstractTargetFunction, params::AbstractVector{<:Real}) =
    ExecCompat(false, 0)

exec_compat(::typeof(target_logval!), target::AbstractTargetFunction, params::AbstractMatrix{<:Real}) =
    ExecCompat(false, 0) # Will need to change when default implementation of target_logval! gets support for parallel execution



struct ConstTargetFunction{T<:Real} <: AbstractTargetFunction
    log_value::T
end

export ConstTargetFunction

function target_logval(
    target::ConstTargetFunction,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    target.log_value
end


# function target_logval!(
#     r::AbstractArray{<:Real},
#     target::ConstTargetFunction,
#     params::AbstractMatrix{<:Real},
#     exec_context::ExecContext
# )
#     @assert size(params, 2) == length(r)
#     fill!(r, target.log_value)
# end





struct GenericTargetFunction{F} <: AbstractTargetFunction
    log_f::F
end

export GenericTargetFunction

function target_logval(
    target::GenericTargetFunction,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    target.log_f(params)
end



struct MvDistTargetFunction{D<:Distribution{Multivariate,Continuous}} <: AbstractTargetFunction
    d::D
end

export MvDistTargetFunction


function target_logval(
    target::MvDistTargetFunction,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    Distributions.logpdf(target.d, params)
end


function target_logval!(
    r::AbstractArray{<:Real},
    target::MvDistTargetFunction,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext = ExecContext()
)
    # TODO: Parallel execution, depending on exec_context
    Distributions.logpdf!(r, target.d, params)
end


exec_compat(::typeof(target_logval!), target::MvDistTargetFunction, params::AbstractMatrix{<:Real}) =
    ExecCompat(false, 0) # Will need to change when implementation of target_logval! gets support for parallel execution



struct GenericProductTargetFunction{T<:Real,P<:Real}
    log_terms::Vector{FunctionWrapper{T,Tuple{P}}}
    single_exec_compat::ExecCompat
end

export GenericProductTargetFunction


function target_logval(
    target::GenericProductTargetFunction{T,P},
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
) where {T,P}
    # TODO: Use exec_context and target.exec_compat
    sum((log_term(convert(P, T)) for (log_term, p) in (target.log_terms, params)))
end


exec_compat(::typeof(target_logval), target::GenericProductTargetFunction, params::AbstractVector{<:Real}) =
    target.single_exec_compat


# Add: Product of target functions
