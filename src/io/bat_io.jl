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
"""
function bat_read end
export bat_read

function bat_read(src::Union{AbstractString,Tuple{AbstractString, AbstractString}})
    ftype = _file_type_from_extension(src)
    if ftype == "HDF5"
        _h5io_open(src, "r") do input
            _h5io_read(input)
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
end
