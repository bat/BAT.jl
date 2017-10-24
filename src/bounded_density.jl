# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct BoundedDensity{
    F<:AbstractDensityFunction,
    B<:AbstractParamBounds
} #<: AbstractBoundedDensity
    density::F
    bounds::B
end

export BoundedDensity

Base.parent(density::BoundedDensity) = density.density

param_bounds(density::BoundedDensity) = density.bounds
nparams(density::BoundedDensity) = nparams(density.bounds)



@inline density_logval(density::BoundedDensity, args...) =
    density_logval(parent(density), args...)

@inline exec_capabilities(::typeof(density_logval), density::BoundedDensity, args...) =
    exec_capabilities(density_logval, parent(density), args...)


@inline density_logval!(density::BoundedDensity, args...) =
    density_logval(parent(density), args...)

@inline exec_capabilities(::typeof(density_logval!), density::BoundedDensity, args...) =
    exec_capabilities(density_logval, parent(density), args...)
