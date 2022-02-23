# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Implementation(s) provided in separate file, loaded via require:
function _h5io_keys end
function _h5io_objtype end
function _h5io_open end


function _h5io_add_path_to_dest(dest_with_subpath::Tuple{Any,AbstractString}, path::AbstractString)
    dest, old_path = dest_with_subpath
    @assert !isempty(old_path)
    @assert !isempty(path)
    (dest, "$old_path/$path")
end

_h5io_add_path_to_dest(dest, path::AbstractString) = (dest, path)


function _h5io_write!(dest_with_subpath::Tuple{Any,AbstractString}, data::AbstractArray{<:Real})
    dest, path = dest_with_subpath
    dest[path] = data
    nothing
end


function _h5io_write!(dest_with_subpath::Tuple{Any,AbstractString}, data::AbstractArray{<:Nothing})
    nothing
end


function _h5io_write!(dest_with_subpath::Tuple{Any,AbstractString}, data::VectorOfSimilarVectors)
    _h5io_write!(dest_with_subpath, Array(flatview(data)))
end


function _h5io_write!(dest, data::NamedTuple{names}) where {names}
    for name in names
        _h5io_write!(_h5io_add_path_to_dest(dest, String(name)), Base.getproperty(data, name))
    end
    nothing
end

function _h5io_write!(dest, data::FillArrays.Fill{<:Real,1})
    _h5io_write!(dest, convert(Array, data))
end

function _h5io_write!(dest, data::FillArrays.Fill{<:AbstractVector{<:Real},1})
    _h5io_write!(dest, convert(VectorOfSimilarVectors, data))
end

function _h5io_write!(dest, data)
    Tables.istable(data) || throw(ArgumentError("data is not a table."))
    cols = Tables.columns(data)::NamedTuple
    @assert cols isa NamedTuple
    _h5io_write!(dest, cols)
end


function _h5io_read end


function _h5io_read(src_with_subpath::Tuple{Any,AbstractString})
    src, subpath = src_with_subpath
    _h5io_read(src[subpath])
end

_h5io_read(src) = _h5io_read(src, _h5io_objtype(src))

_h5io_read(src, ::Val{:dataset}) = _h5io_read_postprocess(read(src))

function _h5io_read(src, ::Val{:datafile})
    names = (sort(_h5io_keys(src))...,)
    values = map(k -> _h5io_read((src, k)), names)
    name_symbols = map(Symbol, names)
    x = NamedTuple{name_symbols}(values)
    _h5io_read_postprocess(x)
end


_h5io_read_postprocess(V::AbstractVector{<:Real}) = V

_h5io_read_postprocess(VV::AbstractMatrix{<:Real}) = VectorOfSimilarVectors(VV)

_h5io_read_postprocess(nt::NamedTuple) = TypedTables.Table(nt)

function _h5io_read_postprocess(nt::NamedTuple{(:info, :logd, :v, :weight)})
    DensitySampleVector((
        _h5io_read_postprocess_samples(nt.v),
        nt.logd, nt.weight, nt.info, Array{Nothing}(undef, size(nt.info)...))
    )
end

# Column :info will be missing if `eltype(samples.info)` was `Nothing`:
function _h5io_read_postprocess(nt::NamedTuple{(:logd, :v, :weight)})
    _h5io_read_postprocess(merge((info = Array{Nothing}(undef, size(nt.weight)...),), nt))
end

_h5io_read_postprocess(nt::NamedTuple{(:chaincycle, :chainid, :sampletype, :stepno)}) =
    MCMCSampleIDVector((nt.chainid, nt.chaincycle, nt.stepno, nt.sampletype))


function _const_col_value(col::AbstractVector)
    r = first(col)
    all(isequal(r), col) ? r : missing
end

_normalize_vs(shape::AbstractValueShape) = shape

function _normalize_vs(shape::NamedTupleShape{names,AT,VT}) where {names,AT,VT}
    NamedTupleShape(VT, map(_normalize_vs, (;shape...)))
end

_normalize_vs(shape::ScalarShape{<:Real}) = ScalarShape{Real}()
_normalize_vs(shape::ArrayShape{<:Real}) = ArrayShape{Real}(size(shape)...)

function _infer_vs_from_table(v::TypedTables.Table)
    cols = Tables.columns(v)
    raw_vs = map(valshape, map(first, cols))
    const_v = map(_const_col_value, cols)
    NamedTupleShape(map((c, shp) -> ismissing(c) ? _normalize_vs(shp) : ConstValueShape(copy(deepcopy(c))), const_v, (;raw_vs...)))
end


_h5io_read_postprocess_samples(v::AbstractVector) = v

function _h5io_read_postprocess_samples(v::TypedTables.Table)
    shp = _infer_vs_from_table(v)
    unshaped_v = VectorOfSimilarVectors(unshaped.(v, Ref(shp)))
    shp.(unshaped_v)
end
