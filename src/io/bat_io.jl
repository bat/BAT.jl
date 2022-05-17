# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function _file_type_from_extension(filename::AbstractString)
    if endswith(filename, ".h5") || endswith(filename, ".hdf5")
        "HDF5"
    else
        throw(ArgumentError("Unknown file type of file \"$filename\""))
    end
end

function _file_type_from_extension(fn_with_subpath::Tuple{AbstractString, AbstractString})
    filename, subpath = fn_with_subpath
    _file_type_from_extension(filename)
end


"""
    bat_read(filename::AbstractString, data)
    bat_read(fn_with_subpath::Tuple{AbstractString, AbstractString}, data)

Read data from a file `filename`, resp. from an internal sub-path of the file
(if supported by the file format), e.g. an HDF5 group.

Currently supported file formats are:

* HDF5 with file extension ".h5" or ".hdf5"

!!! note

    HDF5 I/O functionality is only available when the
    [HDF5](https://github.com/JuliaIO/HDF5.jl) package is loaded (e.g. via
    `import HDF5`).
"""
function bat_read end
export bat_read

function bat_read(src::Union{AbstractString,Tuple{AbstractString, AbstractString}})
    ftype = _file_type_from_extension(src)
    if ftype == "HDF5"
        _h5io_open(src, "r") do input
            #=
                Currently (HDF5.jl - v0.16.9), the keyword `track_order` is ignored in read-in. 
                Thus, HDF5.IDX_TYPE[] has to be set manually.
                The try-catch-block is necessary in order to be able to load old files.
            =#
            prev = HDF5.IDX_TYPE[] 
            HDF5.IDX_TYPE[] = HDF5.API.H5_INDEX_CRT_ORDER
            r = try
                (result = _h5io_read(input),)
            catch err
                HDF5.IDX_TYPE[] = prev
                (result = _h5io_read(input),)
            end
            HDF5.IDX_TYPE[] = prev
            r            
        end
    else
        throw(ArgumentError("Unknown file type $ftype"))
    end
end


"""
    bat_write(filename::AbstractString, data)
    bat_write(fn_with_subpath::Tuple{AbstractString, AbstractString}, data)

Write data to a file `filename`, resp. to an internal sub-path of the file
(if supported by the file format), e.g. an HDF5 group.

Currently supported file formats are:

* HDF5 with file extension ".h5" or ".hdf5"

!!! note

    HDF5 I/O functionality is only available when the
    [HDF5](https://github.com/JuliaIO/HDF5.jl) package is loaded (e.g. via
    `import HDF5`).
"""
function bat_write end
export bat_write

function bat_write(dest::Union{AbstractString,Tuple{AbstractString, AbstractString}}, data)
    ftype = _file_type_from_extension(dest)
    if ftype == "HDF5"
        _h5io_open(dest, "w") do output
            _h5io_write!(output, data)
        end
    else
        throw(ArgumentError("Unknown file type $ftype"))
    end
    nothing
end
