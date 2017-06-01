# This file is a part of BAT.jl, licensed under the MIT License (MIT).


using Compat
#=

@compat abstract type AbstractTargetFunction{
    U<:Real, # Return element type
    V<:AbstractVector{U}, # Return Type
    B<:AbstractParamBounds, # Param bounds
    Diff # Differentiation
} end


immutable TargetFunction{
    U<:Real,
    V<:AbstractVector{U},
    B<:AbstractParamBounds,
    Diff
} <: AbstractTargetFunction{T,U,V,B,Diff}
    
end

Base.ndims(target::TargetFunction)

=#

#=
call_target_function(f::Any, params::AbstractVector) = f(params)
call_target_function(f!::AbstractTargetFunction, params::AbstractVector, aux_values::AbstractVector) = f!(aux_values, params)
=#


#=

Most generic target function type could look like this:

    abstract type AbstractTargetFunction{
        T<:Real, # Required param vector element type - necessary?
        U<:Real, # Return element type
        V<:AbstractVector{U}, # Return Type
        Diff # Differentiation
    } end

`Diff` could be `Val{true|false}` to indicate differentiation support, or either Nothing or a function to be applied to transform the target function.

Non-abstract subtypes (e.g. `SomeTargetFunction <: AbstractTargetFunction{T,U,V}`) would be required to implement

    (f::SomeTargetFunction)(x::AbstractVector{T})::V

To ease typical use cases, BAT-2 should provide a type like

    immutable BoundTargetFunction{N<:Integer, F<:AbstractTargetFunction, X:<AbstractVector} <: AbstractVector
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




