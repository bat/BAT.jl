# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using IntervalSets


export AbstractParamBounds

abstract type AbstractParamBounds{T<:Real} end

Base.eltype{T}(b::AbstractParamBounds{T}) = T



oob{T<:AbstractFloat}(x::T) = T(NaN)
oob{T<:AbstractFloat}(x::T) = typemax(T)

isoob(x) = x == oob(x)

inbounds_or_invalid(x, bounds::ClosedInterval) = iforelse(x in bounds, x, oob(x))


export UnboundedParams

struct UnboundedParams{T<:Real} <: AbstractParamBounds{T}
    ndims::Int
end

Base.length(b::UnboundedParams) = b.ndims

Base.in(params::AbstractVector, bounds::UnboundedParams) = true
Base.in(params::AbstractMatrix, bounds::UnboundedParams, i::Integer) = true



export BoundedParams

abstract type BoundedParams{T<:Real} <: AbstractParamBounds{T} end



export HyperCubeBounds

struct HyperCubeBounds{T<:Real} <: BoundedParams{T}
    from::Vector{T}
    to::Vector{T}

    function HyperCubeBounds{T}(from::Vector{T}, to::Vector{T}) where {T<:Real}
        (indices(from) != indices(to)) && throw(ArgumentError("from and to must have the same indices"))
        @inbounds for i in eachindex(from, to)
            (from[i] > to[i]) && throw(ArgumentError("from[$i] must be <= to[$i]"))
        end
    end
end


HyperCubeBounds{T<:Real}(from::Vector{T}, to::Vector{T}) = HyperCubeBounds{T}(from, to)



Base.length(b::HyperCubeBounds) = length(b.from)


Base.in(params::AbstractVector, bounds::HyperCubeBounds) =
    _multi_array_le(bounds.from, params, bounds.to)

function Base.in(params::AbstractMatrix, bounds::HyperCubeBounds, j::Integer)
    from = bounds.from
    to = bounds.to
    @inbounds for i in eachindex(a,b,c)
        (from[i] <= params[i, j] <= to[i]) || return false
    end
    return true
end



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
