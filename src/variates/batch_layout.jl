# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# BAT stores batches of variates in the layout that `rand(μ^n)` produces:
# vectors of numbers for scalar variates, nested arrays (ArraysOfArrays)
# for array variates and struct vectors for named tuple variates. Constant
# components carry no degrees of freedom and are stored as `Fill` columns.
# Flat storage is the special case of a batch of real vectors.


# Kinds of the columns (leaves) of a batch that are not struct vectors.
# Constants come first, a Fill of numbers is a vector of numbers as well:
struct _ScalarLeaf end
struct _ArrayLeaf end
struct _ConstLeaf end

_leafkind(X::AbstractVector) = _nonconst_leafkind(X)
_leafkind(::AbstractFill{<:Any,1}) = _ConstLeaf()

_nonconst_leafkind(::AbstractVector{<:Real}) = _ScalarLeaf()
_nonconst_leafkind(::AbstractVectorOfSimilarArrays{<:Real}) = _ArrayLeaf()
_nonconst_leafkind(X::AbstractVector) = throw(ArgumentError("Unsupported batch layout of type $(nameof(typeof(X))) with elements of type $(eltype(X))"))


_map_leaves(f, X::StructVector{<:NamedTuple}) = map(C -> _map_leaves(f, C), StructArrays.components(X))
_map_leaves(f, X::AbstractVector) = f(X)

_leaves(X::StructVector{<:NamedTuple}) = _concat_tuples(map(_leaves, values(StructArrays.components(X))))
_leaves(X::AbstractVector) = (X,)

_concat_tuples(xs::Tuple) = isempty(xs) ? () : (first(xs)..., _concat_tuples(Base.tail(xs))...)


# Batches of variates in the layout of BAT's sample storage, from batches
# in that layout or from plain vectors of variates:
_canonical_batch(X::AbstractVector) = X
_canonical_batch(X::AbstractVector{<:AbstractArray}) = convert(VectorOfSimilarArrays, X)
_canonical_batch(X::AbstractVectorOfSimilarArrays) = X
_canonical_batch(X::AbstractSlices) = convert(VectorOfSimilarArrays, X)
_canonical_batch(X::AbstractFill{<:AbstractArray,1}) = X
_canonical_batch(X::AbstractFill{<:NamedTuple,1}) = X
_canonical_batch(X::AbstractVector{<:NamedTuple}) = _canonical_nt_batch(X)
_canonical_nt_batch(X::AbstractVector{<:NamedTuple}) = _canonical_batch(StructArray(X))
# Empty named tuples carry no degrees of freedom:
_canonical_nt_batch(X::AbstractVector{NamedTuple{(),Tuple{}}}) = Fill(NamedTuple(), length(eachindex(X)))
_canonical_batch(X::StructVector{<:NamedTuple}) = StructArray(map(_canonical_batch, StructArrays.components(X)))
_canonical_batch(X::ShapedAsNTArray{<:NamedTuple,1}) = unflatbatch(elshape(X), flatview(unshaped.(X)))
_canonical_batch(X::AbstractVector{<:ShapedAsNT}) = _canonical_batch(map(getindex, X))


"""
    BAT.batch_varshape(X::AbstractVector)

*BAT-internal, not part of stable public API.*

The shape of the variates in the batch `X`, derived from its layout.
`Fill` columns have constant value shapes.
"""
function batch_varshape end

batch_varshape(X::StructVector{<:NamedTuple}) = NamedTupleShape(map(batch_varshape, StructArrays.components(X)))
batch_varshape(X::AbstractVector) = _leaf_varshape(_leafkind(X), X)

_leaf_varshape(::_ScalarLeaf, ::AbstractVector) = ScalarShape{Real}()
_leaf_varshape(::_ArrayLeaf, X::AbstractVector) = ArrayShape{Real}(innersize(X)...)
_leaf_varshape(::_ConstLeaf, X::AbstractVector) = ConstValueShape(getindex_value(X))


"""
    BAT.flatbatch(X::AbstractVector)::AbstractMatrix{<:Real}

*BAT-internal, not part of stable public API.*

The free components of the variates in the batch `X` as the columns of a
matrix, in the standard-variate layout of their shape: leaves in
declaration order, nested named tuples depth-first, arrays column-major,
constants left out.

Returns the flat storage of `X` itself for batches of real vectors and a
new array, on the compute unit of `X`, otherwise. See
[`BAT.unflatbatch`](@ref) for the inverse.
"""
function flatbatch end

flatbatch(X::AbstractVectorOfSimilarVectors{<:Real}) = flatview(X)

function flatbatch(X::AbstractVector)
    blocks = _flat_blocks(X)
    n = length(eachindex(X))
    m = sum(B -> size(B, 1), blocks, init = 0)
    T = isempty(blocks) ? Float64 : mapreduce(eltype, promote_type, blocks)
    M = allocate_array(_flat_alloc_unit(get_compute_unit(X)), T, (m, n))
    offset = 0
    for B in blocks
        k = size(B, 1)
        view(M, (offset + 1):(offset + k), :) .= B
        offset += k
    end
    return M
end

_flat_blocks(X::AbstractVector) = _concat_tuples(map(_flat_block, _leaves(X)))

_flat_block(X::AbstractVector) = _flat_block(_leafkind(X), X)
_flat_block(::_ScalarLeaf, X::AbstractVector) = (reshape(X, 1, :),)
_flat_block(::_ArrayLeaf, X::AbstractVector) = (reshape(flatview(X), :, length(eachindex(X))),)
_flat_block(::_ConstLeaf, ::AbstractVector) = ()

_flat_alloc_unit(cunit::AbstractComputeUnit) = cunit
_flat_alloc_unit(::HeterogeneousComputing.ComputeUnitIndependent) = CPUnit()


"""
    BAT.unflatbatch(vs::AbstractValueShape, M::AbstractMatrix{<:Real})
    BAT.unflatbatch(μ::AbstractMeasure, M::AbstractMatrix{<:Real})

*BAT-internal, not part of stable public API.*

The batch of variates of shape `vs`, resp. of the variate shape of `μ`,
whose free components are the columns of `M` (see [`BAT.flatbatch`](@ref)).
Constant components become `Fill` columns. Copies the contents of `M`.
"""
function unflatbatch end

unflatbatch(μ::AbstractMeasure, M::AbstractMatrix{<:Real}) = unflatbatch(varshape(μ), M)

function unflatbatch(vs::AbstractValueShape, M::AbstractMatrix{<:Real})
    size(M, 1) == totalndof(vs) || throw(DimensionMismatch("Variate shape $vs has $(totalndof(vs)) degrees of freedom, but flat batch has $(size(M, 1)) rows"))
    return _unflat_leaf(vs, M, 0)
end

# Leaves are copied into new plain arrays on the compute unit of `M`:
function _unflat_leaf(::ScalarShape, M::AbstractMatrix, offset::Int)
    X = allocate_array(_flat_alloc_unit(get_compute_unit(M)), eltype(M), (size(M, 2),))
    X .= vec(view(M, _flat_rows(M, offset, 1), :))
    return X
end

function _unflat_leaf(vs::ArrayShape, M::AbstractMatrix, offset::Int)
    flat = allocate_array(_flat_alloc_unit(get_compute_unit(M)), eltype(M), (size(vs)..., size(M, 2)))
    reshape(flat, :, size(M, 2)) .= view(M, _flat_rows(M, offset, totalndof(vs)), :)
    return VectorOfSimilarArrays(flat)
end

_flat_rows(M::AbstractMatrix, offset::Int, len::Int) = firstindex(M, 1) .+ (offset:(offset + len - 1))

_unflat_leaf(vs::ConstValueShape, M::AbstractMatrix, ::Int) = Fill(vs.value, size(M, 2))

function _unflat_leaf(vs::NamedTupleShape, M::AbstractMatrix, offset::Int)
    columns = map(acc -> _unflat_leaf(acc.shape, M, offset + acc.offset), values(vs))
    return StructArray(NamedTuple{keys(vs)}(columns))
end


# A single variate of shape `vs` from per-component values `r` of its free
# components in standard-variate layout (e.g. per-component statistics),
# with `constf(value)` for constant components:
_unflat_variate(vs::AbstractValueShape, r::AbstractVector, constf) = _unflat_variate(vs, r, constf, 0)
_unflat_variate(::ScalarShape, r::AbstractVector, ::Any, offset::Int) = r[offset + 1]
_unflat_variate(vs::ArrayShape, r::AbstractVector, ::Any, offset::Int) = reshape(r[(offset + 1):(offset + totalndof(vs))], size(vs))
_unflat_variate(vs::ConstValueShape, ::AbstractVector, constf, ::Int) = constf(vs.value)

function _unflat_variate(vs::NamedTupleShape, r::AbstractVector, constf, offset::Int)
    NamedTuple{keys(vs)}(map(acc -> _unflat_variate(acc.shape, r, constf, offset + acc.offset), values(vs)))
end


# The batch `X` with variates of shape `vs`, reshaped via the flat layout if
# the variates of `X` have a different shape:
_batch_with_shape(X::AbstractVector, vs::AbstractValueShape) =
    batch_varshape(X) <= vs ? X : unflatbatch(vs, flatbatch(X))


"""
    BAT._own_batch(X::AbstractVector, μ::AbstractMeasure)

*BAT-internal, not part of stable public API.*

The batch `X` of variates of `μ` in the layout of BAT's sample storage:
constant components (point masses) of `μ` become `Fill` columns and lazy
columns are materialized. Dense columns are taken as they are, `rand` and
`transports` return fresh storage. Plain vectors of variates are converted
to that layout.
"""
_own_batch(X::AbstractVector, μ::AbstractMeasure) = _own_batch_impl(_canonical_batch(X), varshape(μ))

function _own_batch_impl(X::StructVector{<:NamedTuple}, vs::NamedTupleShape)
    propertynames(X) == keys(vs) || throw(ArgumentError("Batch with fields $(propertynames(X)) doesn't match variate shape $vs"))
    columns = map((acc, C) -> _own_batch_impl(C, acc.shape), values(vs), values(StructArrays.components(X)))
    return StructArray(NamedTuple{keys(vs)}(columns))
end

_own_batch_impl(X::AbstractVector, vs::ConstValueShape) = Fill(vs.value, length(eachindex(X)))

function _own_batch_impl(X::AbstractVector, vs::AbstractValueShape)
    vs isa NamedTupleShape && throw(ArgumentError("Batch of type $(nameof(typeof(X))) doesn't match variate shape $vs"))
    return _dense_batch(X)
end

_dense_batch(X::StructVector{<:NamedTuple}) = StructArray(map(_dense_batch, StructArrays.components(X)))
_dense_batch(X::AbstractVector) = _dense_leaf(_leafkind(X), X)

_dense_leaf(::_ConstLeaf, X::AbstractVector) = X
_dense_leaf(::_ScalarLeaf, X::AbstractVector) = X isa DenseArray ? X : copy(X)

function _dense_leaf(::_ArrayLeaf, X::AbstractVector)
    flat = flatview(X)
    flat isa DenseArray ? X : VectorOfSimilarArrays(copy(flat))
end


"""
    BAT._vcat_batches(Xs::AbstractVector...)

*BAT-internal, not part of stable public API.*

Concatenate batches of variates column by column. `Fill` columns stay
`Fill` columns if their values are equal and are materialized otherwise.
"""
function _vcat_batches end

function _vcat_batches(X::StructVector{<:NamedTuple}, Xs::StructVector{<:NamedTuple}...)
    names = propertynames(X)
    all(Y -> propertynames(Y) == names, Xs) || throw(ArgumentError("Can't concatenate batches of variates with different fields"))
    columns = map(k -> _vcat_batches(getproperty(X, k), map(Y -> getproperty(Y, k), Xs)...), names)
    return StructArray(NamedTuple{names}(columns))
end

function _vcat_batches(X::AbstractFill{<:Any,1}, Xs::AbstractFill{<:Any,1}...)
    value = getindex_value(X)
    if all(Y -> isequal(getindex_value(Y), value), Xs)
        return Fill(value, sum(Y -> length(eachindex(Y)), (X, Xs...)))
    else
        return _vcat_dense(map(_materialize_leaf, (X, Xs...))...)
    end
end

_vcat_batches(X::AbstractVector, Xs::AbstractVector...) = _vcat_dense(map(_materialize_leaf, (X, Xs...))...)

_materialize_leaf(X::AbstractVector) = X
_materialize_leaf(X::AbstractFill{<:Real,1}) = collect(X)

function _materialize_leaf(X::AbstractFill{<:AbstractArray{<:Real},1})
    value = getindex_value(X)
    flat = similar(value, size(value)..., length(eachindex(X)))
    flat .= value
    return VectorOfSimilarArrays(flat)
end

_vcat_dense(Xs::AbstractVector{<:Real}...) = vcat(Xs...)

function _vcat_dense(Xs::AbstractVectorOfSimilarArrays{<:Real,N}...) where N
    sz = innersize(first(Xs))
    all(Y -> innersize(Y) == sz, Xs) || throw(DimensionMismatch("Can't concatenate batches of arrays of different sizes"))
    return VectorOfSimilarArrays(cat(map(flatview, Xs)..., dims = N + 1))
end

_vcat_dense(Xs::AbstractVector...) = throw(ArgumentError("Can't concatenate batches of variates with different layouts"))


# Leafwise comparison of batches:

_batches_equal(cmp, X::StructVector{<:NamedTuple}, Y::StructVector{<:NamedTuple}) =
    propertynames(X) == propertynames(Y) &&
    all(k -> _batches_equal(cmp, getproperty(X, k), getproperty(Y, k)), propertynames(X))

_batches_equal(cmp, X::AbstractVector, Y::AbstractVector) = cmp(X, Y)


# Batches are grown in place only in flat storage (e.g. MCMC output in
# standard space):
_is_growable(X::StructVector{<:NamedTuple}) = all(_is_growable, values(StructArrays.components(X)))
_is_growable(::Vector) = true
_is_growable(X::AbstractVectorOfSimilarArrays) = flatview(X) isa ElasticArray
_is_growable(::AbstractVector) = false

function _check_growable(X::AbstractVector)
    _is_growable(X) || throw(ArgumentError("Can't grow batches of variates of type $(nameof(typeof(X))) in place, use vcat instead"))
    return nothing
end
