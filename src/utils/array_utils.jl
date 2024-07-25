# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_iscontiguous(A::Array) = true
_iscontiguous(A::AbstractArray) = Base.iscontiguous(A)


_car_cdr_impl() = ()
_car_cdr_impl(x, y...) = (x, (y...,))
_car_cdr(tp::Tuple) = _car_cdr_impl(tp...)


"""
    @propagate_inbounds sum_first_dim(A::AbstractArray, j::Integer, ks::Integer...)

*BAT-internal, not part of stable public API.*

Calculate the equivalent of `sum(A[:, j, ks...])`.
"""
Base.@propagate_inbounds function sum_first_dim(A::AbstractArray, j::Integer, ks::Integer...)
    s = zero(eltype(A))
    @boundscheck if !Base.checkbounds_indices(Bool, Base.tail(axes(A)), (j, ks...))
        throw(BoundsError(A, (:, j)))
    end
    @inbounds for i in axes(A, 1)
        s += A[i, j, ks...]
    end
    s
end


"""
    @propagate_inbounds sum_first_dim(A::AbstractArray)

*BAT-internal, not part of stable public API.*

If `A` is a vector, return `sum(A)`, else `sum(A, 1)[:]`.
"""
sum_first_dim(A::AbstractArray) = sum(A, 1)[:]
sum_first_dim(A::AbstractVector) = sum(A)


const SingleArrayIndex = Union{Integer, CartesianIndex}


convert_numtype(::Type{T}, x::T) where {T<:Real} = x
convert_numtype(::Type{T}, x::Real) where {T<:Real} = convert(T, x)
convert_numtype(::Type{T}, x::Integer) where {T<:AbstractFloat} = x

convert_numtype(::Type{T}, x::AbstractArray{T}) where {T<:Real} = x
convert_numtype(::Type{T}, x::AbstractArray{<:Real}) where {T<:Real} = convert.(T, x)

convert_numtype(::Type{T}, x::Tuple) where {T<:Real} = map(Base.Fix1(convert_numtype, T), x)
convert_numtype(::Type{T}, x::NamedTuple{names}) where {T<:Real, names} = NamedTuple{names}(convert_numtype(T, values(x)))

convert_numtype(::Type{T}, x::ArrayOfSimilarArrays{T}) where {T<:Real} = x
convert_numtype(::Type{T}, x::ArrayOfSimilarArrays{<:Real,M,N}) where {T<:Real,M,N} =
    ArrayOfSimilarArrays{T,M,N}(convert_numtype(T, flatview(x)))

convert_numtype(::Type{T}, x::ShapedAsNT{<:Any,<:AbstractArray{T}}) where {T<:Real} = x
convert_numtype(::Type{T}, x::ShapedAsNT{<:Any,<:AbstractArray{<:Real}}) where {T<:Real} =
    valshape(x)(convert_numtype(T, unshaped(x)))

convert_numtype(::Type{T}, x::ShapedAsNTArray{<:Any,N,<:AbstractArray{<:AbstractArray{T}}}) where {T<:Real,N} = x
convert_numtype(::Type{T}, x::ShapedAsNTArray{<:Any,N,<:AbstractArray{<:AbstractArray{<:Real}}}) where {T<:Real,N} =
    elshape(x).(convert_numtype(T, unshaped.(x)))


any_isinf(trg_v::Real) = isinf(trg_v)
any_isinf(trg_v::AbstractVector{<:Real}) = any(isinf, trg_v)


# Similar to ForwardDiffPullbacks._fieldvals:

object_contents(x::Tuple) = x
object_contents(x::AbstractArray) = x
object_contents(x::NamedTuple) = values(x)

@generated function object_contents(x)
    accessors = [:(getfield(x, $i)) for i in 1:fieldcount(x)]
    :(($(accessors...),))
end


function gen_adapt(gen::GenContext, x)
    cunit = get_compute_unit(gen)
    T = get_precision(gen)
    adapt(cunit, convert_numtype(T, x))
end


const _IntWeightType = Int
const _FloatWeightType = Float64
