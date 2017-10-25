# This file is a part of BAT.jl, licensed under the MIT License (MIT).


doc"""
    log_volume(vol::SpatialVolume)

Get the logarithm of the volume of the space in `vol`.
"""
function log_volume end





doc"""
    fromuhc!(Y::VecOrMat, X::VecOrMat, vol::SpatialVolume)

Bijective transformation of coordinates `X` within the unit hypercube to
coordinates `Y` in `vol`. If `X` and `Y` are matrices, the transformation is
applied to the column vectors.

Use

    inv(fromui)

to get the the inverse transformation.
"""
function fromuhc! end
export fromuhc!


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


doc"""
    from_unit_hypercube!(Y::VecOrMat, X::VecOrMat, vol::SpatialVolume)
    y = from_unit_hypercube(x::Real, lo_hi::ClosedInterval{<:Real})

Map coordinates `X` from the unit hypercube to coordinates `Y` in

"""
function from_unit_hypercube end
export from_unit_hypercube


@inline function from_unit_hypercube!(y::Real, x::Real, HyperRectVolume)
    _all_in_ui
    @boundscheck x in 0..1 || throw(ArgumentError("Input value not in 0..1"))
    muladd(x, (hi - lo), lo)
end




abstract type SpatialVolume{T<:Real} end

export SpatialVolume

Base.eltype(b::SpatialVolume{T}) where T = T

Base.rand(rng::AbstractRNG, vol::SpatialVolume) =
    rand!(rng, vol, Vector{float(eltype(vol))}(ndims(vol)))

Base.rand(rng::AbstractRNG, vol::SpatialVolume, n::Integer) =
    rand!(rng, vol, Matrix{float(eltype(vol))}(ndims(vol), n))



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

HyperRectVolume{T<:Real}(lo::Vector{T}, hi::Vector{T}) = HyperRectVolume{T}(lo, hi)

Base.in(x::AbstractVector, vol::HyperRectVolume) =
    _all_lteq(vol.lo, x, vol.hi)

# ToDo:
# Base.in(x::Matrix, vol::HyperRectVolume) =


function Base.in(X::AbstractMatrix, vol::HyperRectVolume, j::Integer)
    lo = vol.lo
    hi = vol.hi
    for i in indices(X, 1)
        (lo[i] <= X[i, j] <= hi[i]) || return false
    end
    return true
end

Base.ndims(vol::HyperRectVolume) = length(vol.lo)

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
        s += JuliaLibm.log(R(hi[i]) - R(lo[i]))
    end
    R(s)
end
