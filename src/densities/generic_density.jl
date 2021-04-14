# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    GenericDensity{F<:Function} <: AbstractDensity

Constructors:

    GenericDensity(f)

Turns the density function `f` into a BAT-compatible [`AbstractDensity`](@ref).
The return type of `f(v)` must supported by [`logvalof`](@ref).

It must be safe to execute `f` in parallel on multiple threads and
processes.
"""
struct GenericDensity{F<:Function} <: AbstractDensity
    f::F
end

Base.convert(::Type{GenericDensity}, f::Function) = GenericDensity(f)
Base.convert(::Type{AbstractDensity}, f::Function) = GenericDensity(f)


Base.parent(density::GenericDensity) = density.f


function eval_logval_unchecked(density::GenericDensity, v::Any)
    logvalof(density.f(v))
end



"""
    BAT.LogFuncDensity{F<:Function} <: AbstractDensity

*BAT-internal, not part of stable public API.*

Constructors:

    LogFuncDensity(logf::Function)

A density defined by a function that computes it's logarithmic value at given
points.

It must be safe to execute `f` in parallel on multiple threads and
processes.
"""
struct LogFuncDensity{F<:Function} <: AbstractDensity
    logf::F
end

Base.convert(::Type{LogFuncDensity}, nt::NamedTuple{(:logdensity,)}) = LogFuncDensity(nt.logdensity)
Base.convert(::Type{AbstractDensity}, nt::NamedTuple{(:logdensity,)}) = convert(LogFuncDensity, nt)

function eval_logval_unchecked(density::LogFuncDensity, v::Any)
    density.logf(v)
end
