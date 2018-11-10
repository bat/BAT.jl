# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc """
    nparams(X::Union{AbstractParamBounds,MCMCIterator,...})

Get the number of parameters of `X`.
"""
function nparams end
export nparams



export AbstractParamBounds

abstract type AbstractParamBounds end


function Base.intersect(a::AbstractParamBounds, b::AbstractParamBounds)
    nparams(a) != nparams(b) && throw(ArgumentError("Can't intersect parameter bounds with different number of parameters"))
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
export BoundsType
export hard_bounds, cyclic_bounds, reflective_bounds

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


@doc """
    apply_bounds!(params::AbstractVector, bounds::AbstractParamBounds)

Apply `bounds` to parameters `params`.
"""
function apply_bounds! end


@doc """
    apply_bounds(x::<:Real, lo::<:Real, hi::<:Real, boundary_type::BoundsType)

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

@doc """
    apply_bounds(x::Real, interval::ClosedInterval, boundary_type::BoundsType)

Specify lower and upper bound via `interval`.
"""
@inline apply_bounds(x::Real, interval::ClosedInterval, boundary_type::BoundsType, oobval = oob(x)) =
    apply_bounds(x, minimum(interval), maximum(interval), boundary_type, oobval)



export NoParamBounds

struct NoParamBounds <: AbstractParamBounds
    ndims::Int
end

# TODO: Find better solution to determine param type if no bounds:
Base.eltype(bounds::NoParamBounds) = Float64


Base.in(params::AbstractVector, bounds::NoParamBounds) = true
Base.in(params::VectorOfSimilarVectors, bounds::NoParamBounds, i::Integer) = true

nparams(b::NoParamBounds) = b.ndims

apply_bounds!(params::Union{AbstractVector,VectorOfSimilarVectors}, bounds::NoParamBounds, setoob = true) = params

unsafe_intersect(a::NoParamBounds, b::NoParamBounds) = a
unsafe_intersect(a::AbstractParamBounds, b::NoParamBounds) = a
unsafe_intersect(a::NoParamBounds, b::AbstractParamBounds) = b


abstract type ParamVolumeBounds{T<:Real, V<:SpatialVolume{T}} <: AbstractParamBounds end
export ParamVolumeBounds


Base.in(params::AbstractVector, bounds::ParamVolumeBounds) = in(params, bounds.vol)


# Random.rand(rng::AbstractRNG, bounds::ParamVolumeBounds) =
#     rand!(rng, bounds, Vector{float(eltype(bounds))}(nparams(bounds)))

# Random.rand(rng::AbstractRNG, bounds::ParamVolumeBounds, n::Integer) =
#     rand!(rng, bounds, Matrix{float(eltype(bounds))}(nparams(bounds), n))
#
# Random.rand!(rng::AbstractRNG, bounds::ParamVolumeBounds, x::StridedVecOrMat{<:Real}) = rand!(rng, spatialvolume(bounds), x)


nparams(b::ParamVolumeBounds) = ndims(b.vol)


@doc """
    spatialvolume(b::ParamVolumeBounds)::SpatialVolume

Returns the spatial volume that defines the parameter bounds.
"""
function spatialvolume end
export spatialvolume



struct HyperRectBounds{T<:Real} <: ParamVolumeBounds{T, HyperRectVolume{T}}
    vol::HyperRectVolume{T}
    bt::Vector{BoundsType}

    function HyperRectBounds{T}(vol::HyperRectVolume{T}, bt::Vector{BoundsType}) where {T<:Real}
        axes(bt) != (1:ndims(vol),) && throw(ArgumentError("bt must have indices (1:ndims(vol),)"))
        isempty(vol) && throw(ArgumentError("vol must not be empty"))
        new{T}(vol, bt)
    end
end

export HyperRectBounds

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
        if iv_a in iv_b
            c.bt[i] = a.bt[i]
        elseif iv_b in iv_a
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


spatialvolume(bounds::HyperRectBounds) = bounds.vol

function apply_bounds!(params::Union{AbstractVector,VectorOfSimilarVectors}, bounds::HyperRectBounds, setoob = true)
    params_flat = flatview(params)
    if setoob
        params_flat .= apply_bounds.(flatview(params), bounds.vol.lo, bounds.vol.hi, bounds.bt)
    else
        params_flat .= apply_bounds.(flatview(params), bounds.vol.lo, bounds.vol.hi, bounds.bt, params)
    end
    params
end
