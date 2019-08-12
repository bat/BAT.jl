# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ConstDensity{B<:ParamVolumeBounds,T<:Real} <: AbstractDensity
    bounds::B
    log_value::T
end

export ConstDensity


ConstDensity(bounds::ParamVolumeBounds{T,V}, ::typeof(one))  where {T,V}=
    ConstDensity(bounds, zero(T))

ConstDensity(bounds::ParamVolumeBounds, ::typeof(normalize)) =
    ConstDensity(bounds, -log_volume(spatialvolume(bounds)))


Base.convert(::Type{AbstractDensity}, bounds::ParamVolumeBounds) =
    ConstDensity(bounds, one)


param_bounds(density::ConstDensity) = density.bounds

nparams(density::ConstDensity) = nparams(density.bounds)


function density_logval(
    density::ConstDensity,
    params::AbstractVector{<:Real}
)
    density.log_value
end


Distributions.sampler(density::ConstDensity) = spatialvolume(param_bounds(density))


function Statistics.cov(density::ConstDensity{<:HyperRectBounds})
    vol = spatialvolume(param_bounds(density))
    #flat_var = (vol.hi - vol.lo).^2 / 12
    flat_var = var.(Uniform.(vol.lo, vol.hi))
    Matrix(PDiagMat(flat_var))
end
