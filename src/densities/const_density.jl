# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ConstDensity{B<:VarVolumeBounds,T<:Real} <: DistLikeDensity
    bounds::B
    log_value::T
end


ConstDensity(bounds::VarVolumeBounds{T,V}, ::typeof(one))  where {T,V}=
    ConstDensity(bounds, zero(T))

ConstDensity(bounds::VarVolumeBounds, ::typeof(normalize)) =
    ConstDensity(bounds, -log_volume(spatialvolume(bounds)))


Base.convert(::Type{AbstractDensity}, bounds::VarVolumeBounds) =
    ConstDensity(bounds, one)


function density_logval_impl(
    density::ConstDensity,
    v::AbstractVector{<:Real}
)
    density.log_value
end

var_bounds(density::ConstDensity) = density.bounds

ValueShapes.varshape(density::ConstDensity) = ArrayShape{Real}(totalndof(density.bounds))

Distributions.sampler(density::ConstDensity) = spatialvolume(var_bounds(density))

function Statistics.cov(density::ConstDensity{<:HyperRectBounds})
    vol = spatialvolume(var_bounds(density))
    #flat_var = (vol.hi - vol.lo).^2 / 12
    flat_var = var.(Uniform.(vol.lo, vol.hi))
    Matrix(PDiagMat(flat_var))
end
