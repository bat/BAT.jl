# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function _all_lteq(A::AbstractArray, B::AbstractArray, C::AbstractArray)
    indices(A) == indices(B) == indices(C) || throw(DimensionMismatch("A, B and C must have the same indices"))
    result = 0
    @inbounds @simd for i in eachindex(A, B, C)
        result += ifelse(A[i] <= B[i] <= C[i], 1, 0)
    end
    result == length(linearindices(A))
end


@inline function _all_lteq_impl(a::Real, B::AbstractArray, c::Real)
    result = 0
    @inbounds @simd for b in B
        result += ifelse(a <= b <= c, 1, 0)
    end
    result == length(linearindices(B))
end

_all_lteq(a::Real, B::AbstractArray, c::Real) = _all_lteq_impl(a, B, c)

_all_in_ui(X::AbstractArray) = _all_lteq_impl(0, X, 1)


doc"""
    y = fromui(x::Real, lo::Real, hi::Real)
    y = fromui(x::Real, lo_hi::ClosedInterval{<:Real})

Linear bijective transformation from the unit inverval (i.e. `x ∈ 0..1`) to
`y ∈ lo..hi`.

Use

    inv(fromui)

to get the the inverse transformation.

Use `@inbounds` to disable range checking on the input value.
"""
function fromui end
export fromui

@inline function fromui(x::Real, lo::Real, hi::Real)
    @boundscheck x in 0..1 || throw(ArgumentError("Input value not in 0..1"))
    muladd(x, (hi - lo), lo)
end

Base.@propagate_inbounds fromui(x::Real, lo_hi::ClosedInterval{<:Real}) =
    fromui(x, minimum(lo_hi), maximum(lo_hi))

@inline function inv_fromui(x::Real, lo::Real, hi::Real)
    @boundscheck x in lo..hi || throw(ArgumentError("Input value not in lo..hi"))
    (x - lo) / (hi - lo)
end

Base.@propagate_inbounds inv_fromui(x::Real, lo_hi::ClosedInterval{<:Real}) =
    inv_fromui(x, minimum(lo_hi), maximum(lo_hi))

@inline Base.inv(::typeof(fromui)) = inv_fromui
@inline Base.inv(::typeof(inv_fromui)) = fromui
