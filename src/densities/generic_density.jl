# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc """
    GenericDensity{F<:Function} <: AbstractDensity

Constructors:

    GenericDensity(log_f)

Turns the logarithmic density function `log_f` into a BAT-compatible
`AbstractDensity`. `log_f` must support

    `log_f(params::Any)::Real`

It must be safe to execute `log_f` in parallel on multiple threads and
processes.
"""
struct GenericDensity{F<:Function} <: AbstractDensity
    log_f::F
end

export GenericDensity

Base.convert(::Type{GenericDensity}, log_f::Function) = GenericDensity(log_f)
Base.convert(::Type{AbstractDensity}, log_f::Function) = GenericDensity(log_f)


Base.parent(density::GenericDensity) = density.log_f


function density_logval(
    density::GenericDensity,
    params::Any
)
    density.log_f(params)
end
