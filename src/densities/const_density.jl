# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ConstDensity{B<:Union{VarVolumeBounds,Missing},T<:Real} <: DistLikeDensity
    bounds::B
    log_value::T
end


ConstDensity(bounds::Missing, ::typeof(one)) = ConstDensity(bounds, 0)

ConstDensity(bounds::VarVolumeBounds{T,V}, ::typeof(one))  where {T,V} =
    ConstDensity(bounds, zero(T))

ConstDensity(bounds::VarVolumeBounds, ::typeof(normalize)) =
    ConstDensity(bounds, -log_volume(spatialvolume(bounds)))


Base.convert(::Type{ConstDensity}, bounds::VarVolumeBounds) = ConstDensity(bounds, one)
Base.convert(::Type{AbstractDensity}, bounds::VarVolumeBounds) = convert(ConstDensity, bounds)

Base.convert(::Type{ConstDensity}, value::AbstractDensityValue) = ConstDensity(missing, logvalof(value))
Base.convert(::Type{AbstractDensity}, value::AbstractDensityValue) = convert(ConstDensity, value)


@inline density_logval_impl(density::ConstDensity, v::Any) = density.log_value


var_bounds(density::ConstDensity) = density.bounds


ValueShapes.varshape(density::ConstDensity) = ArrayShape{Real}(totalndof(density.bounds))

ValueShapes.varshape(density::ConstDensity{<:Missing}) = missing


Distributions.sampler(density::ConstDensity) = spatialvolume(var_bounds(density))

function Statistics.cov(density::ConstDensity{<:HyperRectBounds})
    vol = spatialvolume(var_bounds(density))
    #flat_var = (vol.hi - vol.lo).^2 / 12
    flat_var = var.(Uniform.(vol.lo, vol.hi))
    Matrix(PDiagMat(flat_var))
end
