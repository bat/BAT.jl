# This file is a part of BAT.jl, licensed under the MIT License (MIT).





#=


struct TargetFunction{
    T<:Real, # Return type
    P<:Real, # Parameter type
    F<:MultiVarProdFunction,
    B<:AbstractParamBounds,
}
    log_f::F
    param_bounds::B
end



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




