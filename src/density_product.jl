# This file is a part of BAT.jl, licensed under the MIT License (MIT).

XXXXX

mutable struct DensityProduct{
    Normalized,
    HasPrior,
    F1<:AbstractDensityFunction,
    F2<:AbstractDensityFunction
} <: AbstractDensityFunction
    d1::F1
    d2::F2
end

export DensityProduct

Base.parent(density::DensityProduct) = density.density

param_bounds(density::DensityProduct) = density.bounds
nparams(density::DensityProduct) = nparams(density.bounds)

param_prior(density::DensityProduct) = param_prior(parent(density))



function @inline density_logval(density::DensityProduct, args...)
    log_x = density_logval(density.d1, args...)
    convert(typeof(log_x), log_x + density.log_norm_factor)
end

function @inline exec_capabilities(::typeof(density_logval), density::DensityProduct, args...)
    exec_capabilities(density_logval, density.d1, args...) +
        exec_capabilities(density_logval, density.d2, args...)
end


function @inline density_logval!(density::DensityProduct, args...)
    log_xs = density_logval(parent(density), args...)
    log_xs .+= density.log_norm_factor
    log_xs
end

@inline exec_capabilities(::typeof(density_logval!), density::DensityProduct, args...) =
    exec_capabilities(density_logval, parent(density), args...)
