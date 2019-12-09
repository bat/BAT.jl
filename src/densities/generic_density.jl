# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    GenericDensity{F<:Function} <: AbstractDensity

*BAT-internal, not part of stable public API.*

Constructors:

    GenericDensity(log_f)

Turns the logarithmic density function `log_f` into a BAT-compatible
`AbstractDensity`. `log_f` must support

    `log_f(v::Any)::Real`

It must be safe to execute `log_f` in parallel on multiple threads and
processes.
"""
struct GenericDensity{F<:Function} <: AbstractDensity
    log_f::F
end

Base.convert(::Type{GenericDensity}, log_f::Function) = GenericDensity(log_f)
Base.convert(::Type{AbstractDensity}, log_f::Function) = GenericDensity(log_f)


Base.parent(density::GenericDensity) = density.log_f


function density_logval(density::GenericDensity, v::Any)
    density.log_f(v)
end
