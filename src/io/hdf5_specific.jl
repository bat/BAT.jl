# This file is a part of BAT.jl, licensed under the MIT License (MIT).


_h5io_keys(df::HDF5.H5DataStore) = keys(df)
_h5io_keys(df::HDF5.Dataset) = nothing

_h5io_objtype(df::HDF5.H5DataStore) = Val(:datafile)
_h5io_objtype(df::HDF5.Dataset) = Val(:dataset)

# Should be available from HDF5 v0.16.3 
_h5_track_order_available() = isdefined(HDF5, :class_getproperty) && 
    isdefined(HDF5, :FileCreateProperties) && 
    in(:track_order, HDF5.class_propertynames(HDF5.FileCreateProperties)) 

_h5open(args...) = _h5_track_order_available() ? HDF5.h5open(args...; track_order = true) : HDF5.h5open(args...)


function _h5io_open(body::Function, filename::AbstractString, mode::AbstractString)
    _h5open(filename, mode) do f
        body(f)        
    end
end


function _h5io_open(body::Function, fn_with_subpath::Tuple{AbstractString, AbstractString}, mode::AbstractString)
    filename, subpath = fn_with_subpath
    _h5open(filename, mode) do f
        body((f, subpath))     
    end
end


"""
    bat_read(src::HDF5.H5DataStore)
    bat_read(src_with_subpath::Tuple{HDF5.H5DataStore, AbstractString})

Read data from HDF5 file or group `src` (optionally from an HDF5-path
relative to `src`).
"""
function bat_read(dest)
    #=
        Currently (HDF5.jl - v0.16.9), the keyword `track_order` is ignored in read-in. 
        Thus, HDF5.IDX_TYPE[] has to be set manually.
        The try-catch-block is necessary in order to be able to load old files.
    =#
    if _h5_track_order_available()
        prev = HDF5.IDX_TYPE[] 
        HDF5.IDX_TYPE[] = HDF5.API.H5_INDEX_CRT_ORDER
        r = try
            _h5io_read(dest)
        catch err
            HDF5.IDX_TYPE[] = prev
            _h5io_read(dest)
        end
        HDF5.IDX_TYPE[] = prev
        r
    else
        _h5io_read(dest)
    end
end


"""
    bat_write(dest::HDF5.H5DataStore, data)
    bat_write(dest_with_subpath::Tuple{HDF5.H5DataStore, AbstractString}, data)

Write `data` to HDF5 file or group `dest` (optionally to an HDF5-path
relative to `dest`).

`data` must be a table (i.e. implement the Tables.jl API).
"""
function bat_write(dest, data)
    _h5io_write!(dest, data)
end
