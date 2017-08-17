# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using IntervalSets


export AbstractParamBounds

abstract type AbstractParamBounds{T<:Real} end

Base.eltype{T}(b::AbstractParamBounds{T}) = T



@inline oob{T<:AbstractFloat}(::Type{T}) = T(NaN)
@inline oob{T<:Integer}(::Type{T}) = typemax(T)
@inline oob(x::Real) = oob(typeof(x))

@inline isoob(x) = x == oob(x)

@inline inbounds_or_invalid(x, bounds::ClosedInterval) = iforelse(x in bounds, x, oob(x))


@enum BoundsType hard_bounds=1 cyclic_bounds=2 reflective_bounds=3
export BoundsType
export hard_bounds
export cyclic_bounds
export reflective_bounds


@inline float_iseven(n::T) where {T<:AbstractFloat} = (n - T(2) * floor((n + T(0.5)) * T(0.5))) < T(0.5)

@inline function apply_bounds(x::X, lo::L, hi::H, boundary_type::BoundsType) where {X<:Real,L<:Real,H<:Real}
    T = float(promote_type(X, L, H))

    offs = ifelse(x < lo, lo - x, x - hi)
    hi_lo = hi - lo
    nwrapped = floor(offs / (hi_lo))
    even_nwrapped = float_iseven(nwrapped)
    wrapped_offs = muladd(-nwrapped, (hi_lo), offs)

    hb = (boundary_type == hard_bounds)
    rb = (boundary_type == reflective_bounds)

    ifelse(
        lo <= x <= hi,
        convert(T, x),
        ifelse(
            hb,
            oob(T),
            ifelse(
                (x < lo && (!rb || rb && !even_nwrapped)) || (x > lo && rb && even_nwrapped),
                convert(T, hi - wrapped_offs),
                convert(T, lo + wrapped_offs)
            )
        )
    )
end

@inline apply_bounds(x::Real, interval::ClosedInterval, boundary_type::BoundsType) =
    apply_bounds(x, minimum(interval), maximum(interval), boundary_type)



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
    bt::Vector{BoundsType}

    function HyperCubeBounds{T}(from::Vector{T}, to::Vector{T}, bt::Vector{BoundsType}) where {T<:Real}
        (indices(from) != indices(to)) && throw(ArgumentError("from and to must have the same indices"))
        @inbounds for i in eachindex(from, to)
            (from[i] > to[i]) && throw(ArgumentError("from[$i] must be <= to[$i]"))
        end
        new{T}(from, to, bt)
    end
end


HyperCubeBounds{T<:Real}(from::Vector{T}, to::Vector{T}, bt::Vector{BoundsType}) = HyperCubeBounds{T}(from, to, bt)



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
