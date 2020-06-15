# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const _default_PT = Float32 # Default type/eltype for variate/parameter values


abstract type AbstractVarBounds end


function Base.intersect(a::AbstractVarBounds, b::AbstractVarBounds)
    totalndof(a) != totalndof(b) && throw(ArgumentError("Can't intersect bounds with different number of DOF"))
    unsafe_intersect(a, b)
end

@inline oob(::Type{T}) where {T<:AbstractFloat} = T(NaN)
@inline oob(::Type{T}) where {T<:Integer} = typemax(T)
@inline oob(x::Real) = oob(typeof(x))

@inline isoob(x::AbstractFloat) = isnan(x)
@inline isoob(x::Integer) = x == oob(x)
isoob(xs::AbstractVector) = any(isoob, xs)
# isoob(xs::VectorOfSimilarVectors) = any(isoob, flatview(xs))


@enum BoundsType hard_bounds=1 cyclic_bounds=2 reflective_bounds=3

function Base.intersect(a::BoundsType, b::BoundsType)
    if a == hard_bounds || b == hard_bounds
        hard_bounds
    elseif a == reflective_bounds || b == reflective_bounds
        reflective_bounds
    else
        cyclic_bounds
    end
end



@inline float_iseven(n::T) where {T<:AbstractFloat} = (n - T(2) * floor((n + T(0.5)) * T(0.5))) < T(0.5)


@doc doc"""
    apply_bounds!(x::AbstractVector, bounds::AbstractVarBounds)

*BAT-internal, not part of stable public API.*

Apply `bounds` to variate/parameters `x`.
"""
function apply_bounds! end


@doc doc"""
    apply_bounds(x::<:Real, lo::<:Real, hi::<:Real, boundary_type::BoundsType)

*BAT-internal, not part of stable public API.*

Apply lower/upper bound `lo`/`hi` to value `x`. `boundary_type` may be
`hard_bounds`, `cyclic_bounds` or `reflective_bounds`.
"""
@inline function apply_bounds(x::X, lo::L, hi::H, boundary_type::BoundsType, oobval = oob(x)) where {X<:Real,L<:Real,H<:Real}
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
            convert(T, oobval),
            ifelse(
                (x < lo && (!rb || rb && !even_nwrapped)) || (x > lo && rb && even_nwrapped),
                convert(T, hi - wrapped_offs),
                convert(T, lo + wrapped_offs)
            )
        )
    )
end

@doc doc"""
    apply_bounds(x::Real, interval::ClosedInterval, boundary_type::BoundsType)

*BAT-internal, not part of stable public API.*

Specify lower and upper bound via `interval`.
"""
@inline apply_bounds(x::Real, interval::ClosedInterval, boundary_type::BoundsType, oobval = oob(x)) =
    apply_bounds(x, minimum(interval), maximum(interval), boundary_type, oobval)



struct NoVarBounds <: AbstractVarBounds
    ndims::Int
end

# TODO: Find better solution to determine value type if no bounds:
Base.eltype(bounds::NoVarBounds) = _default_PT


Base.in(x::AbstractVector, bounds::NoVarBounds) = true
# Base.in(x::VectorOfSimilarVectors, bounds::NoVarBounds, i::Integer) = true

ValueShapes.totalndof(b::NoVarBounds) = b.ndims

apply_bounds!(x::Union{AbstractVector,VectorOfSimilarVectors}, bounds::NoVarBounds, setoob = true) = x

unsafe_intersect(a::NoVarBounds, b::NoVarBounds) = a
unsafe_intersect(a::AbstractVarBounds, b::NoVarBounds) = a
unsafe_intersect(a::NoVarBounds, b::AbstractVarBounds) = b


abstract type VarVolumeBounds{T<:Real, V<:SpatialVolume{T}} <: AbstractVarBounds end


Base.in(x::AbstractVector, bounds::VarVolumeBounds) = in(x, bounds.vol)


# Random.rand(rng::AbstractRNG, bounds::VarVolumeBounds) =
#     rand!(rng, bounds, Vector{float(eltype(bounds))}(totalndof(bounds)))

# Random.rand(rng::AbstractRNG, bounds::VarVolumeBounds, n::Integer) =
#     rand!(rng, bounds, Matrix{float(eltype(bounds))}(npartotalndofams(bounds), n))
#
# Random.rand!(rng::AbstractRNG, bounds::VarVolumeBounds, x::StridedVecOrMat{<:Real}) = rand!(rng, spatialvolume(bounds), x)


ValueShapes.totalndof(b::VarVolumeBounds) = ndims(b.vol)


@doc doc"""
    spatialvolume(b::VarVolumeBounds)::SpatialVolume

*BAT-internal, not part of stable public API.*

Returns the spatial volume that defines the variate/parameter bounds.
"""
function spatialvolume end



struct HyperRectBounds{T<:Real} <: VarVolumeBounds{T, HyperRectVolume{T}}
    vol::HyperRectVolume{T}
    bt::Vector{BoundsType}

    function HyperRectBounds{T}(vol::HyperRectVolume{T}, bt::AbstractVector{BoundsType}) where {T<:Real}
        nd = ndims(vol)
        axes(bt) != (1:nd,) && throw(ArgumentError("bt must have indices (1:ndims(vol),)"))
        (nd > 0) && isempty(vol) && throw(ArgumentError("Cannot create bounds with emtpy volume of $nd dimensions"))
        new{T}(vol, bt)
    end
end


HyperRectBounds(vol::HyperRectVolume{T}, bt::AbstractVector{BoundsType}) where {T<:Real} = HyperRectBounds{T}(vol, bt)
HyperRectBounds(lo::AbstractVector{T}, hi::AbstractVector{T}, bt::AbstractVector{BoundsType}) where {T<:Real} = HyperRectBounds(HyperRectVolume(lo, hi), bt)
HyperRectBounds(lo::AbstractVector{T}, hi::AbstractVector{T}, bt::BoundsType) where {T<:Real} = HyperRectBounds(lo, hi, fill(bt, size(lo, 1)))
HyperRectBounds(intervals::AbstractVector{<:ClosedInterval{<:Real}}, bt) = HyperRectBounds(minimum.(intervals), maximum.(intervals), bt)

Base.similar(bounds::HyperRectBounds) = HyperRectBounds(
        HyperRectVolume(
            fill!(similar(bounds.vol.lo), zero(eltype(bounds.vol.lo))),
            fill!(similar(bounds.vol.hi), one(eltype(bounds.vol.hi)))
        ),
        similar(bounds.bt)
    )

Base.eltype(bounds::HyperRectBounds{T}) where {T} = T


function Base.intersect(a::HyperRectBounds, b::HyperRectBounds)
    c = similar(a)
    for i in eachindex(a.vol.lo, a.vol.hi, a.bt, b.vol.lo, b.vol.hi, b.bt)
        iv_a = a.vol.lo[i]..a.vol.hi[i]
        iv_b = b.vol.lo[i]..b.vol.hi[i]
        if issubset(iv_a, iv_b)
            c.bt[i] = a.bt[i]
        elseif issubset(iv_b, iv_a)
            c.bt[i] = b.bt[i]
        else
            c.bt[i] = a.bt[i] ∩ b.bt[i]
        end

        iv_c = iv_a ∩ iv_b
        c.vol.lo[i] = minimum(iv_c)
        c.vol.hi[i] = maximum(iv_c)
    end
    c
end


function Base.vcat(xs::HyperRectBounds...)
    lo = vcat(map(x -> x.vol.lo, xs)...)
    hi = vcat(map(x -> x.vol.hi, xs)...)
    bt = vcat(map(x -> x.bt, xs)...)
    HyperRectBounds(lo, hi, bt)
end


spatialvolume(bounds::HyperRectBounds) = bounds.vol

function apply_bounds!(x::Union{AbstractVector,VectorOfSimilarVectors}, bounds::HyperRectBounds, setoob = true)
    x_flat = flatview(x)
    if setoob
        x_flat .= apply_bounds.(flatview(x), bounds.vol.lo, bounds.vol.hi, bounds.bt)
    else
        x_flat .= apply_bounds.(flatview(x), bounds.vol.lo, bounds.vol.hi, bounds.bt, x)
    end
    x
end
