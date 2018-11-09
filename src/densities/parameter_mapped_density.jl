# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ParameterMappedDensity{
    D<:AbstractDensity,
    M<:ParameterMapping,
    B<:AbstractParamBounds,
} <: AbstractDensity
    orig_density::D
    parmap::M
    new_bounds::B
end


ParameterMappedDensity(density::AbstractDensity, parmap::ParameterMapping) =
    ParameterMappedDensity(density, parmap, invmap_params(parmap, param_bounds(density)))

export ParameterMappedDensity

import Base.∘
∘(density::AbstractDensity, parmap::ParameterMapping) =
    ParameterMappedDensity(density, parmap)


Base.parent(density::ParameterMappedDensity) = density.orig_density

BAT.param_bounds(density::ParameterMappedDensity) = density.new_bounds

BAT.nparams(density::ParameterMappedDensity) = nparams(density.new_bounds)


function BAT.unsafe_density_logval(density::ParameterMappedDensity, params::AbstractVector{<:Real}, exec_context::ExecContext)
    BAT.unsafe_density_logval(density.orig_density, map_params(density.parmap, params), exec_context)
end

BAT.exec_capabilities(::typeof(BAT.unsafe_density_logval), density::ParameterMappedDensity, params::AbstractVector{<:Real}) =
    BAT.exec_capabilities(density_logval, density.orig_density, map_params(density.parmap, params))


function BAT.unsafe_density_logval!(r::AbstractVector{<:Real}, density::ParameterMappedDensity, params::VectorOfSimilarVectors{<:Real}, exec_context::ExecContext)
    BAT.unsafe_density_logval!(r, density.orig_density, map_params(density.parmap, params), exec_context)

end

BAT.exec_capabilities(::typeof(BAT.unsafe_density_logval!), r::AbstractVector{<:Real}, density::ParameterMappedDensity, params::VectorOfSimilarVectors{<:Real}) =
    BAT.exec_capabilities(density_logval!, r, density.orig_density, map_params(density.parmap, params))


import Base.∘
∘(density::ConstDensity{<:HyperRectBounds}, parmap::ParameterMapping) =
    ConstDensity(invmap_params(parmap, density.bounds), density.log_value)
