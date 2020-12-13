# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const _default_PT = Float32 # Default type/eltype for variate/parameter values


abstract type AbstractVarBounds end


function Base.intersect(a::AbstractVarBounds, b::AbstractVarBounds)
    totalndof(a) != totalndof(b) && throw(ArgumentError("Can't intersect bounds with different number of DOF"))
    unsafe_intersect(a, b)
end



@inline float_iseven(n::T) where {T<:AbstractFloat} = (n - T(2) * floor((n + T(0.5)) * T(0.5))) < T(0.5)



struct NoVarBounds <: AbstractVarBounds
    ndims::Int
end

# TODO: Find better solution to determine value type if no bounds:
Base.eltype(bounds::NoVarBounds) = _default_PT


Base.in(x::Any, bounds::NoVarBounds) = true

Base.isinf(bounds::NoVarBounds) = true

ValueShapes.totalndof(b::NoVarBounds) = b.ndims


unsafe_intersect(a::NoVarBounds, b::NoVarBounds) = a
unsafe_intersect(a::AbstractVarBounds, b::NoVarBounds) = a
unsafe_intersect(a::NoVarBounds, b::AbstractVarBounds) = b


abstract type VarVolumeBounds{T<:Real, V<:SpatialVolume{T}} <: AbstractVarBounds end


Base.in(x::Any, bounds::VarVolumeBounds) = in(x, bounds.vol)


# Random.rand(rng::AbstractRNG, bounds::VarVolumeBounds) =
#     rand!(rng, bounds, Vector{float(eltype(bounds))}(totalndof(bounds)))

# Random.rand(rng::AbstractRNG, bounds::VarVolumeBounds, n::Integer) =
#     rand!(rng, bounds, Matrix{float(eltype(bounds))}(npartotalndofams(bounds), n))
#
# Random.rand!(rng::AbstractRNG, bounds::VarVolumeBounds, x::StridedVecOrMat{<:Real}) = rand!(rng, spatialvolume(bounds), x)


ValueShapes.totalndof(b::VarVolumeBounds) = ndims(b.vol)


"""
    spatialvolume(b::VarVolumeBounds)::SpatialVolume

*BAT-internal, not part of stable public API.*

Returns the spatial volume that defines the variate/parameter bounds.
"""
function spatialvolume end



struct HyperRectBounds{T<:Real} <: VarVolumeBounds{T, HyperRectVolume{T}}
    vol::HyperRectVolume{T}

    function HyperRectBounds{T}(vol::HyperRectVolume{T}) where {T<:Real}
        nd = ndims(vol)
        (nd > 0) && isempty(vol) && throw(ArgumentError("Cannot create bounds with emtpy volume of $nd dimensions"))
        new{T}(vol)
    end
end


HyperRectBounds(vol::HyperRectVolume{T}) where {T<:Real} = HyperRectBounds{T}(vol)
HyperRectBounds(lo::AbstractVector{T}, hi::AbstractVector{T}) where {T<:Real} = HyperRectBounds(HyperRectVolume(lo, hi))
HyperRectBounds(intervals::AbstractVector{<:ClosedInterval{<:Real}}) = HyperRectBounds(minimum.(intervals), maximum.(intervals))
HyperRectBounds(bounds::AbstractInterval) = HyperRectBounds([bounds.left], [bounds.right])

function HyperRectBounds(bounds::Vector{<:AbstractInterval})
    lo = [b.left for b in bounds]
    hi = [b.right for b in bounds]

    return HyperRectBounds(lo, hi)
end


Base.similar(bounds::HyperRectBounds) = HyperRectBounds(
        HyperRectVolume(
            fill!(similar(bounds.vol.lo), zero(eltype(bounds.vol.lo))),
            fill!(similar(bounds.vol.hi), one(eltype(bounds.vol.hi)))
        )
    )

Base.eltype(bounds::HyperRectBounds{T}) where {T} = T


function Base.intersect(a::HyperRectBounds, b::HyperRectBounds)
    c = similar(a)
    for i in eachindex(a.vol.lo, a.vol.hi, b.vol.lo, b.vol.hi)
        iv_a = a.vol.lo[i]..a.vol.hi[i]
        iv_b = b.vol.lo[i]..b.vol.hi[i]
        iv_c = iv_a ∩ iv_b
        c.vol.lo[i] = minimum(iv_c)
        c.vol.hi[i] = maximum(iv_c)
    end
    c
end

Base.isinf(bounds::HyperRectBounds) = isinf(bounds.vol)


function Base.vcat(xs::HyperRectBounds...)
    lo = vcat(map(x -> x.vol.lo, xs)...)
    hi = vcat(map(x -> x.vol.hi, xs)...)
    HyperRectBounds(lo, hi)
end


spatialvolume(bounds::HyperRectBounds) = bounds.vol
