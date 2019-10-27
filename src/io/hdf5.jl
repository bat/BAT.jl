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


function _h5io_write!(dest_with_subpath::Tuple{Any,AbstractString}, data::VectorOfSimilarVectors)
    _h5io_write!(dest_with_subpath, Array(flatview(data)))
end


function _h5io_write!(dest, data::NamedTuple{names}) where {names}
    for name in names
        _h5io_write!(_h5io_add_path_to_dest(dest, String(name)), Base.getproperty(data, name))
    end
    nothing
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

_h5io_read_postprocess(nt::NamedTuple{(:info, :log_posterior, :log_prior, :params, :weight)}) =
    PosteriorSampleVector((nt.params, nt.log_posterior, nt.log_prior, nt.weight, nt.info))

_h5io_read_postprocess(nt::NamedTuple{(:chaincycle, :chainid, :sampletype, :stepno)}) =
    MCMCSampleIDVector((nt.chainid, nt.chaincycle, nt.stepno, nt.sampletype))
