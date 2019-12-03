# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    GenericDensity{F<:Function} <: AbstractDensity

Constructors:

    GenericDensity(f)

Turns the density function `f` into a BAT-compatible [`AbstractDensity)(@ref).
`f(v)` must return either a [`LogDVal`](@ref) (recommended) or a
[`LinDVal`](@ref).

It must be safe to execute `f` in parallel on multiple threads and
processes.
"""
struct GenericDensity{F<:Function} <: AbstractDensity
    f::F
end

Base.convert(::Type{GenericDensity}, f::Function) = GenericDensity(f)
Base.convert(::Type{AbstractDensity}, f::Function) = GenericDensity(f)


Base.parent(density::GenericDensity) = density.f


function density_logval(density::GenericDensity, v::Any)
    density_logval(density.f(v))
end
