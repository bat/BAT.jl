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


param_bounds(density::ConstDensity) = density.bounds

nparams(density::ConstDensity) = nparams(density.bounds)


function unsafe_density_logval(
    density::ConstDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    density.log_value
end

@inline exec_capabilities(::typeof(unsafe_density_logval), density::ConstDensity, args...) =
    ExecCapabilities(0, true, 0, true)


function unsafe_density_logval!(
    r::AbstractArray{<:Real},
    density::ConstDensity,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext
)
    fill!(r, density.log_value)
end

@inline exec_capabilities(::typeof(unsafe_density_logval!), r::AbstractArray{<:Real}, density::ConstDensity, args...) =
    ExecCapabilities(0, true, 0, true)


Distributions.sampler(density::ConstDensity) = spatialvolume(param_bounds(density))
