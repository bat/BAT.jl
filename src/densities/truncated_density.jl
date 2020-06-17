# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct TruncatedDensity{D<:AbstractDensity,B<:VarVolumeBounds} <: AbstractDensity
    density::D
    bounds::B
end

Base.parent(density::TruncatedDensity) = density.density

var_bounds(density::TruncatedDensity) = density.bounds

density_logval(density::TruncatedDensity, v::AbstractVector{<:Real}) =
    density_logval(parent(density), v)
