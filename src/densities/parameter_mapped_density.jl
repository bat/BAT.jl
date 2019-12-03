# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ParameterMappedDensity{
    D<:AbstractDensity,
    M<:VarMapping,
    B<:AbstractVarBounds,
} <: AbstractDensity
    orig_density::D
    parmap::M
    new_bounds::B
end


ParameterMappedDensity(density::AbstractDensity, parmap::VarMapping) =
    ParameterMappedDensity(density, parmap, invmap_params(parmap, var_bounds(density)))


import Base.∘
∘(density::AbstractDensity, parmap::VarMapping) =
    ParameterMappedDensity(density, parmap)


Base.parent(density::ParameterMappedDensity) = density.orig_density

BAT.var_bounds(density::ParameterMappedDensity) = density.new_bounds

ValueShapes.varshape(density::ParameterMappedDensity) = ArrayShape{Real}(totalndof(density.new_bounds))


function BAT.density_logval(density::ParameterMappedDensity, v::AbstractVector{<:Real})
    BAT.density_logval(density.orig_density, map_vars(density.parmap, v))
end


import Base.∘
∘(density::ConstDensity{<:HyperRectBounds}, parmap::VarMapping) =
    ConstDensity(invmap_params(parmap, density.bounds), density.log_value)
