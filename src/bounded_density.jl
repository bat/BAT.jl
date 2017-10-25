# This file is a part of BAT.jl, licensed under the MIT License (MIT).


#==
mutable struct BoundedDensity{    XXXX check
    Normalized,
    HasPrior,
    F<:AbstractDensityFunction{Normalized,<:Any,HasPrior},
    B<:ParamVolumeBounds,
    T<:Real
} <: AbstractDensityFunction{Normalized,true,HasPrior}
    density::F
    bounds::B
    log_norm_factor::T
end

export BoundedDensity

Base.parent(density::BoundedDensity) = density.density

param_bounds(density::BoundedDensity) = density.bounds
nparams(density::BoundedDensity) = nparams(density.bounds)

param_prior(density::BoundedDensity) = param_prior(parent(density))



function @inline density_logval(density::BoundedDensity, args...)
    log_x = density_logval(parent(density), args...)
    convert(typeof(log_x), log_x + density.log_norm_factor)
end

@inline exec_capabilities(::typeof(density_logval), density::BoundedDensity, args...) =
    exec_capabilities(density_logval, parent(density), args...)


function @inline density_logval!(density::BoundedDensity, args...)
    log_xs = density_logval(parent(density), args...)
    log_xs .+= density.log_norm_factor
    log_xs
end

@inline exec_capabilities(::typeof(density_logval!), density::BoundedDensity, args...) =
    exec_capabilities(density_logval, parent(density), args...)
==#
