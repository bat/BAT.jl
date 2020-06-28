# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const _default_PT = Float32 # Default type/eltype for variate/parameter values


abstract type AbstractVarBounds end


function Base.intersect(a::AbstractVarBounds, b::AbstractVarBounds)
    totalndof(a) != totalndof(b) && throw(ArgumentError("Can't intersect bounds with different number of DOF"))
    unsafe_intersect(a, b)
end


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
    renormalize_variate(bounds::AbstractVarBounds, v::Any)

*BAT-internal, not part of stable public API.*

Apply renormalization (if any) inherent `bounds`, to variate/parameters `x`.
The renormalization must preserve phase-space volume locally (determinant
of Jacobian must be one).
"""
function renormalize_variate end


@doc doc"""
    renormalize_variate_impl(x::<:Real, lo::<:Real, hi::<:Real, boundary_type::BoundsType)

*BAT-internal, not part of stable public API.*

Apply lower/upper bound `lo`/`hi` to value `x`. `boundary_type` may be
`hard_bounds`, `cyclic_bounds` or `reflective_bounds`.
"""
@inline function renormalize_variate_impl(x::X, lo::L, hi::H, boundary_type::BoundsType) where {X<:Real,L<:Real,H<:Real}
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
            convert(T, x),
            ifelse(
                (x < lo && (!rb || rb && !even_nwrapped)) || (x > lo && rb && even_nwrapped),
                convert(T, hi - wrapped_offs),
                convert(T, lo + wrapped_offs)
            )
        )
    )
end

@doc doc"""
    renormalize_variate_impl(x::Real, interval::ClosedInterval, boundary_type::BoundsType)

*BAT-internal, not part of stable public API.*

Specify lower and upper bound via `interval`.
"""
@inline renormalize_variate_impl(x::Real, interval::ClosedInterval, boundary_type::BoundsType) =
    renormalize_variate_impl(x, minimum(interval), maximum(interval), boundary_type)



struct NoVarBounds <: AbstractVarBounds
    ndims::Int
end

# TODO: Find better solution to determine value type if no bounds:
Base.eltype(bounds::NoVarBounds) = _default_PT


Base.in(x::Any, bounds::NoVarBounds) = true

Base.isinf(bounds::NoVarBounds) = true

ValueShapes.totalndof(b::NoVarBounds) = b.ndims

renormalize_variate(bounds::NoVarBounds, v::Any) = v

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
HyperRectBounds(bounds::AbstractInterval, bt::BoundsType) = HyperRectBounds([bounds.left], [bounds.right], bt)

function HyperRectBounds(bounds::Vector{<:AbstractInterval}, bt::BoundsType)
    lo = [b.left for b in bounds]
    hi = [b.right for b in bounds]

    return HyperRectBounds(lo, hi, bt)
end


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

Base.isinf(bounds::HyperRectBounds) = isinf(bounds.vol)


function Base.vcat(xs::HyperRectBounds...)
    lo = vcat(map(x -> x.vol.lo, xs)...)
    hi = vcat(map(x -> x.vol.hi, xs)...)
    bt = vcat(map(x -> x.bt, xs)...)
    HyperRectBounds(lo, hi, bt)
end


spatialvolume(bounds::HyperRectBounds) = bounds.vol

function renormalize_variate(bounds::HyperRectBounds, v::AbstractVector{<:Real})
    VT = typeof(v)
    renormalize_variate_impl.(v, bounds.vol.lo, bounds.vol.hi, bounds.bt)
end
