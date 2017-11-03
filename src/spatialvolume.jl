# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type SpatialVolume{T<:Real} end

export SpatialVolume


Base.eltype(b::SpatialVolume{T}) where T = T

Base.rand(rng::AbstractRNG, vol::SpatialVolume) =
    rand!(rng, vol, Vector{float(eltype(vol))}(ndims(vol)))

Base.rand(rng::AbstractRNG, vol::SpatialVolume, n::Integer) =
    rand!(rng, vol, Matrix{float(eltype(vol))}(ndims(vol), n))


doc"""
    log_volume(vol::SpatialVolume)

Get the logarithm of the volume of the space in `vol`.
"""
function log_volume end
export log_volume


doc"""
    fromuhc!(Y::VecOrMat, X::VecOrMat, vol::SpatialVolume)

Bijective transformation of coordinates `X` within the unit hypercube to
coordinates `Y` in `vol`. If `X` and `Y` are matrices, the transformation is
applied to the column vectors. Use `Y === X` to transform in-place.

Use `inv(fromuhc!)` to get the the inverse transformation.
"""
function fromuhc! end
export fromuhc!

function inv_fromuhc! end

Base.inv(::typeof(fromuhc!)) = inv_fromuhc!
Base.inv(::typeof(inv_fromuhc!)) = fromuhc!


doc"""
    fromuhc(X::VecOrMat, vol::SpatialVolume)

Bijective transformation from unit hypercube to `vol`. See `fromuhc!`.

Use `inv(fromuhc)` to get the the inverse transformation.
"""
function fromuhc(X::VecOrMat, vol::SpatialVolume)
    fromuhc!(similar(X), X, vol)
end
export fromuhc

function inv_fromuhc end

Base.inv(::typeof(fromuhc)) = inv_fromuhc
Base.inv(::typeof(inv_fromuhc)) = fromuhc



struct HyperRectVolume{T<:Real} <: SpatialVolume{T}
    # ToDo: Use origin and widths instead of lo and hi, as in GeometryTypes.HyperRectangle?
    lo::Vector{T}
    hi::Vector{T}

    function HyperRectVolume{T}(lo::Vector{T}, hi::Vector{T}) where {T<:Real}
        (indices(lo) != indices(hi)) && throw(ArgumentError("lo and hi must have the same indices"))
        @inbounds for i in eachindex(lo, hi)
            (lo[i] > hi[i]) && throw(ArgumentError("lo[$i] must be <= hi[$i]"))
        end
        new{T}(lo, hi)
    end
end

export HyperRectVolume

HyperRectVolume(lo::Vector{T}, hi::Vector{T}) where {T<:Real} = HyperRectVolume{T}(lo, hi)

Base.ndims(vol::HyperRectVolume) = size(vol.lo, 1)

Base.similar(vol::HyperRectVolume) = HyperRectVolume(similar(vol.a.lo), similar(vol.a.hi))

Base.in(x::AbstractVector, vol::HyperRectVolume) =
    _all_lteq(vol.lo, x, vol.hi)

# TODO: Remove?
# function Base.in(X::AbstractMatrix, vol::HyperRectVolume, j::Integer)
#     lo = vol.lo
#     hi = vol.hi
#     for i in indices(X, 1)
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

function Base.rand!(rng::AbstractRNG, vol::HyperRectVolume, x::StridedVecOrMat{<:Real})
    rand!(rng, x)
    x .= x .* (vol.hi - vol.lo) .+ vol.lo # TODO: Avoid memory allocation
end


function log_volume(vol::HyperRectVolume{T}) where {T}
    R = promote_type(Float64, T)
    S = promote_type(Double{Float64}, T)
    s = zero(S)
    hi = vol.hi
    lo = vol.lo
    @assert indices(hi) == indices(lo)
    @inbounds @simd for i in eachindex(hi)
        d = max(zero(R), R(hi[i]) - R(lo[i]))
        s += JuliaLibm.log(d)
    end
    R(s)
end


# ToDo:
# log_intersect_volume(a::HyperRectVolume{T}, b::HyperRectVolume{T}) where {T} = ...


doc"""
    fromuhc!(Y::VecOrMat, X::VecOrMat, vol::HyperRectVolume)

Linear bijective transformation from unit hypercube to hyper-rectangle `vol`.
"""
function fromuhc!(Y::VecOrMat, X::VecOrMat, vol::HyperRectVolume)
    _all_in_ui(X) || throw(ArgumentError("X not in unit hypercube"))
    Y .= unsafe_fromui.(X, vol.lo, vol.hi)
end

function inv_fromuhc!(Y::VecOrMat, X::VecOrMat, vol::HyperRectVolume)
    X in vol || throw(ArgumentError("X not in vol"))
    Y .= unsafe_inv_fromui.(X, vol.lo, vol.hi)
end
