# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type SpatialVolume{T<:Real} end


Base.eltype(b::SpatialVolume{T}) where T = T

Random.rand(rng::AbstractRNG, vol::SpatialVolume) =
    rand!(rng, vol, Vector{float(eltype(vol))}(undef, ndims(vol)))

Random.rand(rng::AbstractRNG, vol::SpatialVolume, n::Integer) =
    rand!(rng, vol, VectorOfSimilarVectors(Matrix{float(eltype(vol))}(undef, ndims(vol), n)))


@doc doc"""
    log_volume(vol::SpatialVolume)

*BAT-internal, not part of stable public API.*

Get the logarithm of the volume of the space in `vol`.
"""
function log_volume end


@doc doc"""
    fromuhc!(Y::AbstractVector, X::AbstractVector, vol::SpatialVolume)
    fromuhc!(Y::VectorOfSimilarVectors, X::VectorOfSimilarVectors, vol::SpatialVolume)

*BAT-internal, not part of stable public API.*

Bijective transformation of coordinates `X` within the unit hypercube to
coordinates `Y` in `vol`. If `X` and `Y` are matrices, the transformation is
applied to the column vectors. Use `Y === X` to transform in-place.

Use `inv(fromuhc!)` to get the the inverse transformation.
"""
function fromuhc! end

function inv_fromuhc! end

Base.inv(::typeof(fromuhc!)) = inv_fromuhc!
Base.inv(::typeof(inv_fromuhc!)) = fromuhc!


@doc doc"""
    fromuhc(X::AbstractVector, vol::SpatialVolume)
    fromuhc(X::VectorOfSimilarVectors, vol::SpatialVolume)

*BAT-internal, not part of stable public API.*

Bijective transformation from unit hypercube to `vol`. See `fromuhc!`.

Use `inv(fromuhc)` to get the the inverse transformation.
"""
function fromuhc(X::Union{AbstractVector,VectorOfSimilarVectors}, vol::SpatialVolume)
    fromuhc!(similar(X), X, vol)
end

function inv_fromuhc(X::Union{AbstractVector,VectorOfSimilarVectors}, vol::SpatialVolume)
    inv_fromuhc!(similar(X), X, vol)
end

Base.inv(::typeof(fromuhc)) = inv_fromuhc
Base.inv(::typeof(inv_fromuhc)) = fromuhc



struct HyperRectVolume{T<:Real} <: SpatialVolume{T}
    # ToDo: Use origin and widths instead of lo and hi, as in GeometryTypes.HyperRectangle?
    lo::Vector{T}
    hi::Vector{T}

    function HyperRectVolume{T}(lo::AbstractVector{T}, hi::AbstractVector{T}) where {T<:Real}
        (axes(lo) != axes(hi)) && throw(ArgumentError("lo and hi must have the same indices"))
        new{T}(lo, hi)
    end
end


HyperRectVolume(lo::AbstractVector{T}, hi::AbstractVector{T}) where {T<:Real} = HyperRectVolume{T}(lo, hi)

Base.ndims(vol::HyperRectVolume) = size(vol.lo, 1)

function Base.isempty(vol::HyperRectVolume)
        @inbounds for i in eachindex(vol.lo, vol.hi)
             (vol.lo[i] > vol.hi[i]) && return true
        end
        isempty(vol.lo)
end

Base.similar(vol::HyperRectVolume) = HyperRectVolume(similar(vol.lo), similar(vol.hi))

Base.in(x::AbstractVector, vol::HyperRectVolume) =
    _all_lteq(vol.lo, x, vol.hi)

function Base.isinf(vol::HyperRectVolume)
    return (any(isinf.(vol.hi)) || any(isinf.(vol.lo)))
end

function Base.copy!(
    target::HyperRectVolume{T},
    src::HyperRectVolume{T}) where {T<:AbstractFloat}

    p = ndims(src)
    resize!(target.lo, p)
    copyto!(target.lo, src.lo)
    resize!(src.hi, p)
    copyto!(target.hi, src.hi)
    nothing
end

# TODO: Remove?
# function Base.in(X::AbstractMatrix, vol::HyperRectVolume, j::Integer)
#     lo = vol.lo
#     hi = vol.hi
#     for i in axes(X, 1)
#         (lo[i] <= X[i, j] <= hi[i]) || return false
#     end
#     return true
# end

# ToDo:
# Base.in(x::Matrix, vol::HyperRectVolume) =

function Base.intersect(a::HyperRectVolume, b::HyperRectVolume)
    c = similar(a)
    c.lo .= max.(a.lo, b.lo)
    c.hi .= min.(a.hi, b.hi)
    c
end

function Random.rand!(rng::AbstractRNG, vol::HyperRectVolume, x::Union{AbstractVector{<:Real},VectorOfSimilarVectors{<:Real}})
    x_flat = flatview(x)
    rand!(rng, x_flat)
    x_flat .= x_flat .* (vol.hi - vol.lo) .+ vol.lo # TODO: Avoid memory allocation
    x
end


function log_volume(vol::HyperRectVolume{T}) where {T}
    R = promote_type(Float64, T)
    S = promote_type(DoubleFloat{Float64}, T)
    s = zero(S)
    hi = vol.hi
    lo = vol.lo
    @assert axes(hi) == axes(lo)
    @inbounds @simd for i in eachindex(hi)
        d = max(zero(R), R(hi[i]) - R(lo[i]))
        s += log(d)
    end
    R(s)
end


# ToDo:
# log_intersect_volume(a::HyperRectVolume{T}, b::HyperRectVolume{T}) where {T} = ...


fromuhc!(Y::AbstractVector, X::AbstractVector, vol::HyperRectVolume) = _fromuhc_impl!(Y, X, vol)
fromuhc!(Y::VectorOfSimilarVectors, X::VectorOfSimilarVectors, vol::HyperRectVolume) = _fromuhc_impl!(Y, X, vol)

function _fromuhc_impl!(Y::Union{AbstractVector,VectorOfSimilarVectors}, X::Union{AbstractVector,VectorOfSimilarVectors}, vol::HyperRectVolume)
    _all_in_ui(flatview(X)) || throw(ArgumentError("X not in unit hypercube"))
    Y_flat = flatview(Y)
    Y_flat .= unsafe_fromui.(flatview(X), vol.lo, vol.hi)
    Y
end


inv_fromuhc!(Y::AbstractVector, X::AbstractVector, vol::HyperRectVolume) = _inv_fromuhc_impl!(Y, X, vol)
inv_fromuhc!(Y::VectorOfSimilarVectors, X::VectorOfSimilarVectors, vol::HyperRectVolume) = _inv_fromuhc_impl!(Y, X, vol)

function _inv_fromuhc_impl!(Y::Union{AbstractVector,VectorOfSimilarVectors}, X::Union{AbstractVector,VectorOfSimilarVectors}, vol::HyperRectVolume)
    X in vol || throw(ArgumentError("X not in vol"))
    Y_flat = flatview(Y)
    Y_flat .= unsafe_inv_fromui.(flatview(X), vol.lo, vol.hi)
    Y
end
