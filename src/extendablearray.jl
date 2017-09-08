# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_tuple_head_tail(xs::Tuple) = _tuple_head_tail_impl(xs...)
_tuple_head_tail_impl(x, xs...) = x, xs

function _tuple_firsts_last(xs::Tuple)
    a, b = _tuple_head_tail(reverse(xs))
    reverse(b), a
end

@inline _int_tuple(xs::NTuple{N,Integer}) where {N} = map(x -> convert(Int, x), xs)


struct ExtendableArray{T,N,M} <: DenseArray{T,N}
    kernel_size::NTuple{M,Int}
    kernel_length::Int
    data::Vector{T}

    function ExtendableArray{T}(dims::Integer...) where {T}
        kernel_size, size_lastdim = _split_dims(dims)
        kernel_length = prod(kernel_size)
        data = Vector{T}(kernel_length * size_lastdim)
        new{T,length(dims),length(kernel_size)}(kernel_size, kernel_length, data)
    end
end



_split_dims(dims::NTuple{N,Integer}) where {N} = _tuple_firsts_last(_int_tuple(dims))

function _split_resize_dims(A::ExtendableArray, dims::NTuple{N,Integer}) where {N}
    kernel_size, size_lastdim = _split_dims(dims)
    kernel_size != A.kernel_size && throw(ArgumentError("Can only resize last dimension of an ExtendableArray"))
    kernel_size, size_lastdim
end


Base.size(A::ExtendableArray) = (A.kernel_size..., div(length(linearindices(A.data)), A.kernel_length))
Base.getindex(A::ExtendableArray, i::Integer) = getindex(A.data, i)
Base.setindex!(A::ExtendableArray, x, i::Integer) = setindex!(A.data, x, i)
Base.IndexStyle(::ExtendableArray) = IndexLinear()

Base.length(A::ExtendableArray) = length(A.data)
Base.linearindices(A::ExtendableArray) = linearindices(A.data)


function Base.resize!(A::ExtendableArray, dims::Integer...)
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    resize!(A.data, A.kernel_length * size_lastdim)
    A
end


function Base.sizehint!(A::ExtendableArray, dims::Integer...)
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    sizehint!(A.data, A.kernel_length * size_lastdim)
    A
end


function Base.append!(dest::ExtendableArray, src::AbstractArray)
    mod(length(linearindices(src)), dest.kernel_length) != 0 && throw(DimensionMismatch("Can't append, length of source array is incompatible"))
    append!(dest.data, src)
    dest
end


# function Base.push!(dest::ExtendableArray{T,N1}, src::AbstractArray{T,N2}) where {T,N1,N2}
#     size(src) != dest.kernel_size && throw(DimensionMismatch("Can't push, shape of source array is incompatible"))
#     append!(dest, src)
# end


function _copy_impl!(dest::ExtendableArray, args...)
    copy!(dest.data, args...)
    dest
end

Base.copy!(dest::ExtendableArray, doffs::Integer, src::AbstractArray, args...) = _copy_impl!(dest, doffs, src, args...)
Base.copy!(dest::ExtendableArray, src::AbstractArray) = _copy_impl!(dest, src)
