# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc """
    GenericDensity{F} <: AbstractDensity

Constructors:

    GenericDensity(log_f, nparams::Int)

Turns the logarithmic density function `log_f` into a
BAT-compatible `AbstractDensity`. `log_f` must support

    `log_f(params::AbstractVector{<:Real})::Real`

with `length(params) == nparams`.

It must be safe to execute `log_f` in parallel on multiple threads and
processes.
"""
struct GenericDensity{F} <: AbstractDensity
    log_f::F
    nparams::Int
end

export GenericDensity

Base.parent(density::GenericDensity) = density.log_f

param_bounds(density::GenericDensity) = NoParamBounds(density.nparams)

nparams(density::GenericDensity) = density.nparams

function density_logval(
    density::GenericDensity,
    params::AbstractVector{<:Real}
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    density.log_f(params)
end
