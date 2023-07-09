# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type BATIOAlgorithm

Abstract type for density transformation algorithms.
"""
abstract type BATIOAlgorithm end
export BATIOAlgorithm


"""
    bat_write(
        filename::AbstractString,
        content,
        [algorithm::BATIOAlgorithm]
    )::AbstractMeasureOrDensity

Write `content` to file `filename` using `algorithm`.

Example:
```julia
smpls = bat_sample(posterior, ...).result
bat_write("samples.hdf5", smpls)
```

Returns `(result = filename, ...)`

Result properties not listed here are specific to the output `algorithm` and
are not part of the stable public API.

See [`bat_read`](@ref).

Currently supported file formats are:

* HDF5 with file extension ".h5" or ".hdf5"

!!! note

    HDF5 I/O functionality is only available when the
    [HDF5](https://github.com/JuliaIO/HDF5.jl) package is loaded (e.g. via
    `import HDF5`).

!!! note

    Do not add add algorithms to `bat_write`, add algorithms to
    `bat_write_impl` instead.
"""
function bat_write end
export bat_write

function bat_write_impl end


function bat_write(@nospecialize(filename::AbstractString), @nospecialize(content), algorithm::BATIOAlgorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_write_impl(String(filename), content, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_write(@nospecialize(filename::AbstractString), @nospecialize(content)) = bat_write(filename, content, get_batcontext())

function bat_write(@nospecialize(filename::AbstractString), @nospecialize(content), algorithm::BATIOAlgorithm)
    bat_write(filename, content, algorithm, get_batcontext())
end

function bat_write(@nospecialize(filename::AbstractString), @nospecialize(content), context::BATContext)
    algorithm = bat_default_withinfo(bat_write, Val(:algorithm), String(filename), content)
    bat_write(filename, content, algorithm, context)
end


function argchoice_msg(::typeof(bat_write), ::Val{:algorithm}, x::BATIOAlgorithm)
    "Using output algorithm $x"
end



"""
    bat_read(
        filename::AbstractString,
        [key,]
        [algorithm::BATIOAlgorithm]
    )::AbstractMeasureOrDensity

Read data (optionally selected by `key`) from `filename` using `algorithm`.

Example:
```julia
smpls = bat_read("samples.hdf5", smpls).result
```

Returns `(result = content, ...)`

Result properties not listed here are specific to the output `algorithm` and
are not part of the stable public API.

See [`bat_write`](@ref).

Currently supported file formats are:

* HDF5 with file extension ".h5" or ".hdf5"

!!! note

    HDF5 I/O functionality is only available when the
    [HDF5](https://github.com/JuliaIO/HDF5.jl) package is loaded (e.g. via
    `import HDF5`).

!!! note

    Do not add add algorithms to `bat_read`, add algorithms to
    `bat_read_impl` instead.
"""
function bat_read end
export bat_read

function bat_read_impl end


function bat_read(@nospecialize(filename::AbstractString), key, algorithm::BATIOAlgorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_read_impl(String(filename), String(key), algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_read(@nospecialize(filename::AbstractString), key) = bat_read(filename, key, get_batcontext())

function bat_read(@nospecialize(filename::AbstractString), key, algorithm::BATIOAlgorithm)
    bat_read(filename, key, algorithm, get_batcontext())
end

function bat_read(@nospecialize(filename::AbstractString), key, context::BATContext)
    algorithm = bat_default_withinfo(bat_read, Val(:algorithm), String(filename), String(key))
    bat_read(filename, key, algorithm, context)
end


function bat_read(@nospecialize(filename::AbstractString), algorithm::BATIOAlgorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_read_impl(String(filename), algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_read(@nospecialize(filename::AbstractString)) = bat_read(filename, get_batcontext())

function bat_read(@nospecialize(filename::AbstractString), algorithm::BATIOAlgorithm)
    bat_read(filename, algorithm, get_batcontext())
end

function bat_read(@nospecialize(filename::AbstractString), context::BATContext)
    algorithm = bat_default_withinfo(bat_read, Val(:algorithm), String(filename))
    bat_read(filename, algorithm, context)
end


function argchoice_msg(::typeof(bat_read), ::Val{:algorithm}, x::BATIOAlgorithm)
    "Using input algorithm $x"
end



"""
    struct BATHDF5IO <: BATIOAlgorithm

Selects the BAT HDF5 format as the output format.

See [`bat_write`](@ref) and [`bat_read`](@ref).

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct BATHDF5IO <: BATIOAlgorithm
    function BATHDF5IO()
        pkgext(Val(:HDF5))
        new()
    end
end
export BATHDF5IO
