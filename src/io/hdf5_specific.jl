# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_h5io_keys(df::HDF5.DataFile) = HDF5.names(df)
_h5io_keys(df::HDF5.HDF5Dataset) = nothing

_h5io_objtype(df::HDF5.DataFile) = Val(:datafile)
_h5io_objtype(df::HDF5.HDF5Dataset) = Val(:dataset)


function _h5io_open(body::Function, filename::AbstractString, mode::AbstractString)
    HDF5.h5open(filename, mode) do f
        body(f)        
    end
end


function _h5io_open(body::Function, fn_with_subpath::Tuple{AbstractString, AbstractString}, mode::AbstractString)
    filename, subpath = fn_with_subpath
    HDF5.h5open(filename, mode) do f
        body((f, subpath))     
    end
end


"""
    bat_read(src::HDF5.DataFile)
    bat_read(src_with_subpath::Tuple{HDF5.DataFile, AbstractString})

Read data from HDF5 file or group `src` (optionally from an HDF5-path
relative to `src`).
"""
function bat_read(dest)
    _h5io_read(dest)
end


"""
    bat_write(dest::HDF5.DataFile, data)
    bat_write(dest_with_subpath::Tuple{HDF5.DataFile, AbstractString}, data)

Write `data` to HDF5 file or group `dest` (optionally to an HDF5-path
relative to `dest`).

`data` must be a table (i.e. implement the Tables.jl API).
"""
function bat_write(dest, data)
    _h5io_write!(dest, data)
end
