# This file is a part of BAT.jl, licensed under the MIT License (MIT).
@doc doc"""
    TruncatedDensity

Constructor:

    TruncatedDensity(D<:AbstractDensity, B<:VarVolumeBounds)

*BAT-internal, not part of stable public API.*

Density with specified bounds.
"""
struct TruncatedDensity{D<:AbstractDensity, B<:VarVolumeBounds} <: AbstractDensity
    density::D
    bounds::B
end

Base.parent(density::TruncatedDensity) = density.density

var_bounds(density::TruncatedDensity) = density.bounds

density_logval(density::TruncatedDensity, v::AbstractVector{<:Real}) =
    density_logval(parent(density), v)
