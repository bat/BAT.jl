# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using IntervalSets


export AbstractParamBounds

abstract type AbstractParamBounds{T<:Real} end

Base.eltype{T}(b::AbstractParamBounds{T}) = T



@inline oob{T<:AbstractFloat}(::Type{T}) = T(NaN)
@inline oob{T<:Integer}(::Type{T}) = typemax(T)
@inline oob(x::Real) = oob(typeof(x))

@inline isoob(x::AbstractFloat) = isnan(x)
@inline isoob(x::Integer) = x == oob(x)


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

apply_bounds!(params::AbstractVector, bounds::UnboundedParams) = params


export BoundedParams

abstract type BoundedParams{T<:Real} <: AbstractParamBounds{T} end



export HyperCubeBounds

struct HyperCubeBounds{T<:Real} <: BoundedParams{T}
    lo::Vector{T}
    hi::Vector{T}
    bt::Vector{BoundsType}

    function HyperCubeBounds{T}(lo::Vector{T}, hi::Vector{T}, bt::Vector{BoundsType}) where {T<:Real}
        (indices(lo) != indices(hi)) && throw(ArgumentError("lo and hi must have the same indices"))
        @inbounds for i in eachindex(lo, hi)
            (lo[i] > hi[i]) && throw(ArgumentError("lo[$i] must be <= hi[$i]"))
        end
        new{T}(lo, hi, bt)
    end
end


HyperCubeBounds{T<:Real}(lo::Vector{T}, hi::Vector{T}, bt::Vector{BoundsType}) = HyperCubeBounds{T}(lo, hi, bt)



Base.length(b::HyperCubeBounds) = length(b.lo)


Base.in(params::AbstractVector, bounds::HyperCubeBounds) =
    _multi_array_le(bounds.lo, params, bounds.hi)

function Base.in(params::AbstractMatrix, bounds::HyperCubeBounds, j::Integer)
    lo = bounds.lo
    hi = bounds.hi
    @inbounds for i in eachindex(a,b,c)
        (lo[i] <= params[i, j] <= hi[i]) || return false
    end
    return true
end


apply_bounds!(params::AbstractVecOrMat, bounds::HyperCubeBounds) =
    params .= apply_bounds.(params, bounds.lo, bounds.hi, bounds.bt)


function Base.rand!(rng::AbstractRNG, bounds::HyperCubeBounds, x::StridedVecOrMat{<:Real})
    rand!(rng, x)
    x .= x .* (bounds.hi - bounds.lo) .+ bounds.lo # TODO: Avoid memory allocation
end
