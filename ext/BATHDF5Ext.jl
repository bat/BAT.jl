# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATHDF5Ext

using HDF5
using HDF5: H5DataStore

using BAT
using BAT: MCMCSampleIDVector, SampleID

using ArraysOfArrays, FillArrays, Tables, StructArrays, TypedTables
using ValueShapes


const H5StoreWithSubPath = Union{Tuple{<:HDF5.File,<:AbstractString}, Tuple{<:H5DataStore,<:AbstractString}}
const AnyH5Store = Union{H5DataStore, H5StoreWithSubPath}


BAT.pkgext(::Val{:HDF5}) = BAT.PackageExtension{:HDF5}()


#=
# About HDF5 version differences

Since HDF5 v0.16.3 it is possible to track the order 
of groups via the keyword `track_order`. 
However, until v0.16.11 it is ignored in the read-in.
Thus, HDF5.IDX_TYPE[] has to be set manually.
The try-catch-block is necessary in order to be able to load old files.

Since v0.16.11, IDX_TYPE does not exists anymore,
but the read-in now does not ignore the track_order `anymore`.
=#

# _h5_track_order_kw_available: true if HDF5 >= v0.16.3 
_h5_track_order_kw_available() = :track_order in HDF5.class_propertynames(HDF5.FileCreateProperties)
# _h5_track_order_kw_available: true if v0.16.3 <= HDF5 <= 0.16.10
_h5_IDX_TYPE_available() = isdefined(HDF5, :IDX_TYPE)

_h5open(args...) = _h5_track_order_kw_available() ? HDF5.h5open(args...; track_order = true) : HDF5.h5open(args...)



function BAT.bat_write_impl(filename::AbstractString, content, alg::BATHDF5IO, ::BATContext)
    @nospecialize filename, content, alg
    _h5open(filename, "w") do datastore
        _h5io_write(datastore, content)
    end
    return (result = filename,)
end


function _h5io__get_or_create_group(datastore::H5DataStore, groupname::AbstractString)
    @nospecialize datastore, groupname
    if isempty(groupname) || groupname == "."
        HDF5.open_group(datastore, ".")
    elseif haskey(datastore, groupname)
        HDF5.open_group(datastore, groupname)
    else
        if _h5_track_order_kw_available()
            HDF5.create_group(datastore, groupname; track_order = true)
        else
            HDF5.create_group(datastore, groupname)
        end
    end
end


function _h5io_write(datastore::H5DataStore, content::Pair{<:AbstractString,<:Any})
    @nospecialize datastore, content
    path, data = content
    _h5io_write(datastore, path, data)
    _h5io_is_root(path) || _h5io_mark_parent_groups!(datastore, path)
    nothing
end

_h5io_is_root(path::AbstractString) = path in ("", ".", "/")

function _h5io_mark_parent_groups!(datastore::H5DataStore, path::AbstractString)
    @nospecialize datastore, path
    parent = dirname(path)
    if _h5io_is_root(parent)
        _h5io_set_layout!(_h5io__get_or_create_group(datastore, "."), "group")
    else
        _h5io_set_layout!(_h5io__get_or_create_group(datastore, parent), "group")
        _h5io_mark_parent_groups!(datastore, parent)
    end
end

function _h5io_write(datastore::H5DataStore, content)
    @nospecialize datastore, content
    _h5io_write(datastore, ".", content)
end

# See the docstring of BATHDF5IO for the layout of the written objects.

const _h5_format_version = 6

function _h5io_set_layout!(obj, layout::AbstractString)
    @nospecialize obj
    write_attribute(obj, "bat_layout", layout)
    return obj
end

_h5io_set_fields!(obj, names) = write_attribute(obj, "bat_fields", String[String(name) for name in names])


function _h5io_write(datastore::H5DataStore, path::AbstractString, data::NamedTuple)
    @nospecialize datastore, path, data
    group = _h5io__get_or_create_group(datastore, path)
    _h5io_set_layout!(group, "group")
    _h5io_set_fields!(group, keys(data))
    for (name, x) in pairs(data)
        _h5io_write(group, String(name), x)
    end
    nothing
end

function _h5io_write(datastore::H5DataStore, path::AbstractString, samples::DensitySampleVector)
    @nospecialize datastore, path, samples
    group = _h5io__get_or_create_group(datastore, path)
    _h5io_set_layout!(group, "samples")
    write_attribute(group, "bat_format", _h5_format_version)
    # Columns `info` and `aux` are left out if they hold nothing:
    for name in (:v, :logd, :weight, :info, :aux)
        X = getproperty(samples, name)
        X isa AbstractVector{<:Nothing} || _h5io_write(group, String(name), X)
    end
    nothing
end

_h5io_write(datastore::H5DataStore, path::AbstractString, X::AbstractVector) =
    _h5io_write_batch(datastore, path, BAT._canonical_batch(X))

function _h5io_write_batch(datastore::H5DataStore, path::AbstractString, X::StructVector)
    @nospecialize datastore, path, X
    group = _h5io__get_or_create_group(datastore, path)
    columns = StructArrays.components(X)
    _h5io_set_layout!(group, "namedtuple")
    _h5io_set_fields!(group, keys(columns))
    eltype(X) <: SampleID && write_attribute(group, "bat_eltype", String(nameof(eltype(X))))
    for (name, C) in pairs(columns)
        _h5io_write(group, String(name), C)
    end
    nothing
end

_h5io_write_batch(datastore::H5DataStore, path::AbstractString, X::AbstractVector) =
    _h5io_write_leaf(BAT._leafkind(X), datastore, path, X)

_h5io_write_leaf(::BAT._ScalarLeaf, datastore::H5DataStore, path::AbstractString, X::AbstractVector) =
    _h5io_set_layout!(_h5io_write_value(datastore, path, X), "scalar")

_h5io_write_leaf(::BAT._ArrayLeaf, datastore::H5DataStore, path::AbstractString, X::AbstractVector) =
    _h5io_set_layout!(_h5io_write_value(datastore, path, flatview(X)), "array")

function _h5io_write_leaf(::BAT._ConstLeaf, datastore::H5DataStore, path::AbstractString, X::AbstractVector)
    obj = _h5io_set_layout!(_h5io_write_value(datastore, path, FillArrays.getindex_value(X)), "const")
    write_attribute(obj, "bat_length", length(eachindex(X)))
    return obj
end

# Write a single value, return the HDF5 object written:
function _h5io_write_value(datastore::H5DataStore, path::AbstractString, x::Union{Real,AbstractArray{<:Real}})
    @nospecialize datastore, path, x
    group = _h5io__get_or_create_group(datastore, dirname(path))
    # Arrays like Fill, views or device arrays can't be written directly:
    group[basename(path)] = x isa AbstractArray ? convert(Array, x) : x
    return group[basename(path)]
end

function _h5io_write_value(datastore::H5DataStore, path::AbstractString, x::NamedTuple)
    @nospecialize datastore, path, x
    group = _h5io__get_or_create_group(datastore, path)
    _h5io_set_fields!(group, keys(x))
    for (name, value) in pairs(x)
        _h5io_write_value(group, String(name), value)
    end
    return group
end



function BAT.bat_read_impl(filename::AbstractString, alg::BATHDF5IO, context::BATContext)
    @nospecialize filename, alg
    BAT.bat_read_impl(filename, ".", alg, context)
end

function BAT.bat_read_impl(filename::AbstractString, key::AbstractString, alg::BATHDF5IO, ::BATContext)
    @nospecialize filename, key, alg
    h5open(filename, "r") do input
        if _h5_IDX_TYPE_available() 
            prev = HDF5.IDX_TYPE[] 
            HDF5.IDX_TYPE[] = HDF5.API.H5_INDEX_CRT_ORDER
            return try
                (result = _h5io_read(input, key),)
            catch err
                HDF5.IDX_TYPE[] = prev
                (result = _h5io_read(input, key),)
            finally
                HDF5.IDX_TYPE[] = prev
            end
        else
            (result = _h5io_read(input, key),)
        end
    end
end


function _h5io_read(datastore::H5DataStore, path::AbstractString)
    @nospecialize datastore, path
    obj = isempty(path) || path == "." ? datastore : datastore[path]
    _h5io_read(obj)
end

_h5io_has_layout(obj) = haskey(HDF5.attributes(obj), "bat_layout")

_h5io_read(obj::Union{H5DataStore,HDF5.Dataset}) = _h5io_has_layout(obj) ? _h5io_read_layout(obj) : _h5io_read_legacy(obj)


_h5io_read_layout(obj) = _h5io_read_layout(Val(Symbol(read_attribute(obj, "bat_layout"))), obj)

_h5io_read_layout(::Val{layout}, obj) where layout =
    throw(ArgumentError("Unsupported BAT HDF5 layout \"$layout\" of object $(HDF5.name(obj))"))

function _h5io_read_layout(::Val{:samples}, group::H5DataStore)
    format = read_attribute(group, "bat_format")
    format <= _h5_format_version || throw(ArgumentError(
        "Samples in $(HDF5.name(group)) have BAT HDF5 format version $format, this version of BAT supports versions up to $_h5_format_version"
    ))
    v, logd, weight = map(name -> _h5io_read(group[name]), ("v", "logd", "weight"))
    n = length(eachindex(logd))
    info, aux = map(name -> haskey(group, name) ? _h5io_read(group[name]) : fill(nothing, n), ("info", "aux"))
    DensitySampleVector((v, logd, weight, info, aux))
end

_h5io_read_layout(::Val{:group}, group::H5DataStore) = _h5io_read_fields(_h5io_read, group)

_h5io_read_layout(::Val{:namedtuple}, group::H5DataStore) = _h5io_struct_batch(_h5io_struct_eltype(group), _h5io_read_fields(_h5io_read, group))

_h5io_read_layout(::Val{:scalar}, dataset::HDF5.Dataset) = read(dataset)

_h5io_read_layout(::Val{:array}, dataset::HDF5.Dataset) = VectorOfSimilarArrays(read(dataset))

_h5io_read_layout(::Val{:const}, obj) = Fill(_h5io_read_value(obj), read_attribute(obj, "bat_length"))

_h5io_read_value(dataset::HDF5.Dataset) = read(dataset)
_h5io_read_value(group::H5DataStore) = _h5io_read_fields(_h5io_read_value, group)

function _h5io_read_fields(f_read, group::H5DataStore)
    names = haskey(HDF5.attributes(group), "bat_fields") ? read_attribute(group, "bat_fields") : keys(group)
    NamedTuple{(map(Symbol, names)...,)}(map(name -> f_read(group[name]), names))
end

# Struct batches with BAT's sample info types as elements carry the type name:
function _h5io_struct_eltype(group::H5DataStore)
    haskey(HDF5.attributes(group), "bat_eltype") || return nothing
    name = Symbol(read_attribute(group, "bat_eltype"))
    T = isdefined(BAT, name) ? getproperty(BAT, name) : nothing
    return T isa Type && T <: SampleID ? T : nothing
end

_h5io_struct_batch(::Nothing, columns::NamedTuple) = StructArray(columns)
_h5io_struct_batch(::Type{T}, columns::NamedTuple) where T = StructArray{T}(map(name -> columns[name], fieldnames(T)))



# Files written by BAT 5.x carry no layout attributes, their layout is
# inferred from names and values:

_h5io_read_legacy(src::HDF5.Dataset) = read(src)

function _h5io_read_legacy(datastore::H5DataStore)
    names = keys(datastore)
    values = map(k -> _h5io_read_postprocess(_h5io_read_legacy(datastore[k])), names)
    name_symbols = (map(Symbol, names)...,)
    x = NamedTuple{name_symbols}(values)
    _h5io_read_postprocess(x)
end


_h5io_read_postprocess(V::AbstractVector{<:Real}) = V
_h5io_read_postprocess(V::Union{StructVector,Table}) = V

_h5io_read_postprocess(A::AbstractArray{<:Real}) = VectorOfSimilarArrays(A)

_h5io_read_postprocess(nt::NamedTuple) = begin
    TypedTables.Table(nt)
end

function _h5io_read_postprocess(nt::NamedTuple{(:v, :logd, :weight, :info)})
    DensitySampleVector((
        _h5io_read_postprocess_samples(nt.v),
        nt.logd, nt.weight, nt.info, Array{Nothing}(undef, size(nt.info)...))
    )
end
function _h5io_read_postprocess(nt::NamedTuple{(:info, :logd, :v, :weight)})
    # This method is needed for older hdf5 files where `track_order` was not yet used
    DensitySampleVector((
        _h5io_read_postprocess_samples(nt.v),
        nt.logd, nt.weight, nt.info, Array{Nothing}(undef, size(nt.info)...))
    )
end

# Column :info will be missing if `eltype(samples.info)` was `Nothing`:
function _h5io_read_postprocess(nt::NamedTuple{(:v, :logd, :weight)})
    _h5io_read_postprocess(merge(nt, (info = Array{Nothing}(undef, size(nt.weight)...),)))
end
function _h5io_read_postprocess(nt::NamedTuple{(:logd, :v, :weight)})
    # This method is needed for older hdf5 files where `track_order` was not yet used
    _h5io_read_postprocess(merge((info = Array{Nothing}(undef, size(nt.weight)...),), nt))
end


_h5io_read_postprocess(nt::NamedTuple{(:chainid, :walkerid, :chaincycle, :stepno, :proposalid, :sampletype)}) =
    MCMCSampleIDVector((nt.chainid, nt.walkerid, nt.chaincycle, nt.stepno, nt.proposalid, nt.sampletype))

# This method is needed for older hdf5 files where `proposalid` was not yet used
_h5io_read_postprocess(nt::NamedTuple{(:chainid, :walkerid, :chaincycle, :stepno, :sampletype)}) =
    MCMCSampleIDVector((nt.chainid, nt.walkerid, nt.chaincycle, nt.stepno, nt.sampletype))

# This method is needed for older hdf5 files where `walkerid` was not yet used
_h5io_read_postprocess(nt::NamedTuple{(:chainid, :chaincycle, :stepno, :sampletype)}) =
    MCMCSampleIDVector((nt.chainid, nt.chaincycle, nt.stepno, nt.sampletype))

# This method is needed for older hdf5 files where `track_order` was not yet used
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

# Nested named tuples are read back as nested tables:
_infer_col_shape(col::TypedTables.Table) = _infer_vs_from_table(col)

function _infer_col_shape(col::AbstractVector)
    c = _const_col_value(col)
    ismissing(c) ? _normalize_vs(valshape(first(col))) : ConstValueShape(copy(deepcopy(c)))
end

_infer_vs_from_table(v::TypedTables.Table) = NamedTupleShape(map(_infer_col_shape, Tables.columns(v)))


_h5io_read_postprocess_samples(v::AbstractVector) = v

function _h5io_read_postprocess_samples(v::TypedTables.Table)
    shp = _infer_vs_from_table(v)
    unshaped_v = convert(VectorOfSimilarVectors, unshaped.(v, Ref(shp)))
    BAT.unflatbatch(shp, flatview(unshaped_v))
end


end # module BATHDF5Ext
