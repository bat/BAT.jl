# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_all_in_ui(X::AbstractArray) = _all_lteq_impl(0, X, 1)


@doc """
    y = fromui(x::Real, lo::Real, hi::Real)
    y = fromui(x::Real, lo_hi::ClosedInterval{<:Real})

Linear bijective transformation from the unit inverval (i.e. `x ∈ 0..1`) to
`y ∈ lo..hi`.

Use `inv(fromui)` to get the the inverse transformation.

Use `@inbounds` to disable range checking on the input value.
"""
function fromui end
export fromui

@inline unsafe_fromui(x::Real, lo::Real, hi::Real) = muladd(x, (hi - lo), lo)

@inline function fromui(x::Real, lo::Real, hi::Real)
    @boundscheck x in 0..1 || throw(ArgumentError("Input value not in 0..1"))
    unsafe_fromui(x, lo, hi)
end

Base.@propagate_inbounds fromui(x::Real, lo_hi::ClosedInterval{<:Real}) =
    fromui(x, minimum(lo_hi), maximum(lo_hi))

@inline unsafe_inv_fromui(x::Real, lo::Real, hi::Real) = (x - lo) / (hi - lo)

@inline function inv_fromui(x::Real, lo::Real, hi::Real)
    @boundscheck x in lo..hi || throw(ArgumentError("Input value not in lo..hi"))
    unsafe_inv_fromui(x, lo, hi)
end

Base.@propagate_inbounds inv_fromui(x::Real, lo_hi::ClosedInterval{<:Real}) =
    inv_fromui(x, minimum(lo_hi), maximum(lo_hi))

@inline Base.inv(::typeof(fromui)) = inv_fromui
@inline Base.inv(::typeof(inv_fromui)) = fromui
