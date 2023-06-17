# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATHDF5Ext

@static if isdefined(Base, :get_extension)
    using HDF5
    using HDF5: H5DataStore
else
    using ..HDF5
    using ..HDF5: H5DataStore
end

using BAT
using BAT: MCMCSampleIDVector

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



function BAT.bat_write_impl(filename::AbstractString, content, alg::BATHDF5IO)
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
end

function _h5io_write(datastore::H5DataStore, content)
    @nospecialize datastore, content
    _h5io_write(datastore, ".", content)
end

_to_flat_array(A::AbstractArray{<:Real}) = convert(Array, A)
_to_flat_array(A::ArrayOfSimilarArrays) = _to_flat_array(flatview(A))
_to_flat_array(A::AbstractArray{<:AbstractArray{<:Real}}) = _to_flat_array(ArrayOfSimilarArrays(A))

const _AnyRealArrayOrArrays = Union{AbstractArray{<:Real},AbstractArray{<:AbstractArray{<:Real}}}

function _h5io_write(datastore::H5DataStore, path::AbstractString, data::_AnyRealArrayOrArrays)
    @nospecialize datastore, path, data
    group = _h5io__get_or_create_group(datastore, dirname(path))
    # Some array types like Fill can't be written directly, so convert to Array:
    flat_data = _to_flat_array(data)
    group[basename(path)] = flat_data
    nothing
end

function _h5io_write(datastore::H5DataStore, path::AbstractString, data::AbstractArray{<:Nothing})
    @nospecialize datastore, path, data
    nothing
end

function _h5io_write(datastore::H5DataStore, path::AbstractString, data::NamedTuple)
    @nospecialize datastore, path, data
    group = _h5io__get_or_create_group(datastore, path)
    for name in propertynames(data)
        _h5io_write(group, String(name) => Base.getproperty(data, name))
    end
    nothing
end

const AnyTable = Union{AbstractVector{<:NamedTuple},StructArray}

function _h5io_write(datastore::H5DataStore, path::AbstractString, data::AnyTable)
    @nospecialize datastore, path, data
    Tables.istable(data) || throw(ArgumentError("In _h5io_write data must be a table"))
    cols = Tables.columns(data)::NamedTuple
    @assert cols isa NamedTuple
    _h5io_write(datastore, path, cols)
    nothing
end



function BAT.bat_read_impl(filename::AbstractString, alg::BATHDF5IO)
    @nospecialize filename, alg
    BAT.bat_read_impl(filename, ".", alg)
end

function BAT.bat_read_impl(filename::AbstractString, key::AbstractString, alg::BATHDF5IO)
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
    if isempty(path) || path == "."
        _h5io_read(datastore)
    else
        _h5io_read(datastore[path])
    end
end

_h5io_read(src::HDF5.Dataset) = read(src)

function _h5io_read(datastore::H5DataStore)
    names = keys(datastore)
    values = map(k -> _h5io_read_postprocess(_h5io_read(datastore, k)), names)
    name_symbols = (map(Symbol, names)...,)
    x = NamedTuple{name_symbols}(values)
    _h5io_read_postprocess(x)
end


_h5io_read_postprocess(V::AbstractVector{<:Real}) = V
_h5io_read_postprocess(V::Union{StructVector,Table}) = V

_h5io_read_postprocess(VV::AbstractMatrix{<:Real}) = VectorOfSimilarVectors(VV)

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


end # module BATHDF5Ext
