# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using FunctionWrappers
import FunctionWrappers: FunctionWrapper


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


"""
    target_logval!(
        r::AbstractArray{<:Real},
        target::AbstractTargetFunction,
        params::AbstractMatrix{<:Real},
        exec_context::ExecContext
    )

PDF value of `pdist` for transitioning from old to new parameter values for
multiple parameter sets.

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
    exec_context::ExecContext
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
        exec_context::ExecContext
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
    exec_context::ExecContext
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
    exec_context::ExecContext
)
    target.log_f(params)
end



struct GenericProductTargetFunction{T<:Real,P<:Real}
    log_terms::Vector{FunctionWrapper{T,Tuple{P}}}
    single_exec_compat::ExecCompat
end

export GenericProductTargetFunction


function target_logval(
    target::GenericProductTargetFunction{T,P},
    params::AbstractVector{<:Real},
    exec_context::ExecContext
) where {T,P}
    # TODO: Use exec_context and target.exec_compat
    sum((log_term(convert(P, T)) for (log_term, p) in (target.log_terms, params)))
end


exec_compat(::typeof(target_logval), target::GenericProductTargetFunction, params::AbstractVector{<:Real}) =
    target.single_exec_compat



#=



function (target::DummyTargetFunction)(
    values::AbstractVector{T},
    # gradients::AbstractMatrix{T},
    params::AbstractMatrix{P},
    target::AbstractTargetFunction,
    select::AbstractVector{Bool},  # true/false depending on parameters in bounds
    exec_context::ExecContext
) where {T <: AbstractFloat, P <: AbstractFloat}
    @assert size(values, 1) == size(gradients, 2) == size(params, 2)
    values .= -Inf
    nothing
end

=#



#=

struct TargetFunction{F, B<:AbstractParamBounds} <: Function
    log_f::F
    param_bounds::B
end

function (f::TargetFunction)(
    values::AbstractVector{T},
    params::AbstractMatrix{P}
) where {T <: AbstractFloat, P <: AbstractFloat}
    @assert size(values, 1) == size(gradients, 2) == size(params, 2)
    f.log_f(values, params)
end

function (f::TargetFunction)(
    values::AbstractVector{<:T},
    gradients::AbstractMatrix{<:T},
    params::AbstractMatrix{<:P}
) where {T <: AbstractFloat, P <: AbstractFloat}
    @assert size(values, 1) == size(gradients, 2) == size(params, 2)
    @assert size(gradients, 1) == size(params, 1)
    ...
    @inbounds for i in indices(params, 2)
        if in(params, f.param_bounds, i)
            ...
        else
            ...
        end
    end
    ...
    f.log_f(values, gradients, params)
end



# Target function signature without gradients:

((
    values::AbstractVector{T},
    params::AbstractMatrix{P}
) where {T <: AbstractFloat, P <: Real}) -> begin
    @assert size(values, 1) == size(gradients, 2) == size(params, 2)
    nothing
end

# Target function signature with gradients:

((
    values::AbstractVector{<:T},
    gradients::AbstractMatrix{<:T},
    params::AbstractMatrix{<:P}
) where {T <: AbstractFloat, P <: AbstractFloat}) -> begin
    @assert size(values, 1) == size(gradients, 2) == size(params, 2)
    @assert size(gradients, 1) == size(params, 1)
    nothing
end

=#


#=





abstract type MultiVarProdFunction {
    T<:Real, # Return type
    P<:Real, # Parameter type
    Diff # Differentiation
} <: Function end

function (f::MultiVarProdFunction){P<:Real}(params::AbstractVector{P}) =
    f(linearindices(f), params)

Base.checkbounds{RT<:Integer}(f::MultiVarProdFunction, rng::Range{RT}) =
    Base.checkbounds_indices(Bool, (linearindices(f),), (rng,)) || throw(Boundserror(A, rng))


#=
abstract type UniVarProdFunction {
    T<:Real, # Return type
    P<:Real, # Parameter type
    Diff # Differentiation
} <: Function end
=#

struct MultiVarProdFunctionWrapper <: Function {
    T<:Real, # Return type
    P<:Real, # Parameter type
    Diff # Differentiation,
    F
} <: MultiVarProdFunction{T, P, Void}
    f::F
end

Base.linearindices(f::MultiVarProdFunctionWrapper) = Base.OneTo(1)
Base.linearindices(rng::Range{Int}, f::MultiVarProdFunctionWrapper) = Base.OneTo(1)


(f::MultiVarProdFunctionWrapper){P<:Real}(params::AbstractVector{P}) =
    f(params::AbstractVector{P})

function (f::MultiVarProdFunctionWrapper){P<:Real}(Range{Int}, params::AbstractVector{P}) =
    f(params::AbstractVector{P})


checkbounds_prodfunc



struct TargetFunction{
    U<:Real,
    V<:AbstractVector{U},
    B<:AbstractParamBounds,
    Diff
} <: MultiVarProdFunction{T,U,V,B,Diff}
    log_f::F,
    param_bounds::B    
end

Base.ndims(target::TargetFunction)

=#

#=
call_target_function(f::Any, params::AbstractVector) = f(params)
call_target_function(f!::MultiVarProdFunction, params::AbstractVector, aux_values::AbstractVector) = f!(aux_values, params)
=#


#=

Most generic target function type could look like this:

    abstract type MultiVarProdFunction{
        T<:Real, # Required param vector element type - necessary?
        U<:Real, # Return element type
        V<:AbstractVector{U}, # Return Type
        Diff # Differentiation
    } end

`Diff` could be `Val{true|false}` to indicate differentiation support, or either Nothing or a function to be applied to transform the target function.

Non-abstract subtypes (e.g. `SomeTargetFunction <: MultiVarProdFunction{T,U,V}`) would be required to implement

    (f::SomeTargetFunction)(x::AbstractVector{T})::V

To ease typical use cases, BAT-2 should provide a type like

    struct BoundTargetFunction{N<:Integer, F<:MultiVarProdFunction, X:<AbstractVector} <: AbstractVector
        partitions::NTuple{N, Int} # number of partitions in each dimension
        f::F # Target function
        x::X # parameters
    end

that supports target functions partitioned in multiple dimensions which provide an interface

    size(f::SomeTargetFunction, x::X)::NTuple{N, Int}
    (f::SomeTargetFunction)(x::X, idxs::NTuple{N, Int})::Real
    (f::SomeTargetFunction)(x::X, idxs::NTuple{N, UnitRange{Int}})::Real # returns
        # sum over idxs, provide default implementation

=#




