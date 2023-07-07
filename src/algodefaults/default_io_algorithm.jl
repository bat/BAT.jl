# This file is a part of BAT.jl, licensed under the MIT License (MIT).


bat_default(::BATContext, ::typeof(bat_write), ::Val{:algorithm}, @nospecialize(filename::AbstractString), @nospecialize(content)) =
    _io_alg_from_extension(String(filename))

bat_default(::BATContext, ::typeof(bat_read), ::Val{:algorithm}, @nospecialize(filename::AbstractString), @nospecialize(key)) =
    _io_alg_from_extension(String(filename))

bat_default(::BATContext, ::typeof(bat_read), ::Val{:algorithm}, @nospecialize(filename::AbstractString)) =
    _io_alg_from_extension(String(filename))


function _io_alg_from_extension(filename::String)
    if endswith(filename, ".h5") || endswith(filename, ".hdf5")
        BATHDF5IO()
    else
        throw(ArgumentError("Unknown file type of file \"$filename\""))
    end
end
