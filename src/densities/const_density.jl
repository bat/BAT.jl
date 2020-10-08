# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ConstDensity{T<:Real,B<:Union{VarVolumeBounds,Missing}} <: DistLikeDensity
    value::LogDVal{T}
    bounds::B
end


ConstDensity(value::LogDVal) = ConstDensity(value, missing)

ConstDensity(::typeof(normalize), bounds::Union{VarVolumeBounds,Missing} = missing) =
    ConstDensity(LogDVal(-log_volume(spatialvolume(bounds))), bounds)

Base.@deprecate ConstDensity(::typeof(one)) ConstDensity(LogDVal(0))
Base.@deprecate ConstDensity(::typeof(one), ::Missing) ConstDensity(LogDVal(0), missing)
Base.@deprecate ConstDensity(::typeof(one), bounds::VarVolumeBounds) ConstDensity(LogDVal(0), bounds)

Base.convert(::Type{ConstDensity}, value::LogDVal) = ConstDensity(value)
Base.convert(::Type{AbstractDensity}, value::LogDVal) = convert(ConstDensity, value)


@inline eval_logval_unchecked(density::ConstDensity, v::Any) = logvalof(density.value)


var_bounds(density::ConstDensity) = density.bounds


ValueShapes.varshape(density::ConstDensity{T,<:VarVolumeBounds}) where T = ArrayShape{Real}(totalndof(density.bounds))

ValueShapes.varshape(density::ConstDensity{T,Missing}) where T = missing


Distributions.sampler(density::ConstDensity) = spatialvolume(var_bounds(density))

function Statistics.cov(density::ConstDensity{T,<:HyperRectBounds}) where T
    vol = spatialvolume(var_bounds(density))
    #flat_var = (vol.hi - vol.lo).^2 / 12
    flat_var = var.(Uniform.(vol.lo, vol.hi))
    Matrix(PDiagMat(flat_var))
end
