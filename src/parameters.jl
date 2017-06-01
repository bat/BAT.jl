# This file is a part of BAT.jl, licensed under the MIT License (MIT).


using Compat


export ParamValues

typealias ParamValues{T} StridedVector{T}


export AbstractParamBounds

@compat abstract type AbstractParamBounds{T<:Real} end

Base.eltype{T}(b::AbstractParamBounds{T}) = T



export UnboundedParams

immutable UnboundedParams{T<:Real} <: AbstractParamBounds{T}
    ndims::Int
end

Base.length(b::UnboundedParams) = b.ndims

Base.in(params::AbstractVector, bounds::UnboundedParams) = true



export BoundedParams

@compat abstract type BoundedParams{T<:Real} <: AbstractParamBounds{T} end



export HyperCubeBounds

immutable HyperCubeBounds{T<:Real} <: BoundedParams{T}
    from::Vector{T}
    to::Vector{T}
end

Base.length(b::HyperCubeBounds) = length(b.from)

Base.in(params::AbstractVector, bounds::HyperCubeBounds) =
    _multi_array_le(bounds.from, params, bounds.to)



param_bounds(bounds::AbstractParamBounds, log_f) = (bounds, log_f)

function param_bounds{T}(bounds::Vector{NTuple{2,T}}, log_f)
    U = float(T)
    n = length(bounds)
    from = map!(x -> x[1], Vector{U}(n), bounds)
    to = map!(x -> x[2], Vector{U}(n), bounds)
    (HyperCubeBounds(from, to), log_f)
end

function param_bounds{T}(bounds::NTuple{2,Vector{T}}, log_f)
    length(bounds[1]) != length(bounds[2]) && throw(DimensionMismatch("Lower and upper bound vectors must have the same length"))
    from = float.(bounds[1])
    to = float.(bounds[2])
    (HyperCubeBounds(from, to), log_f)
end
