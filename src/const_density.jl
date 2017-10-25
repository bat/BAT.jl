# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ConstantDensity{T<:Real} <: UnconstrainedDensityFunction
    B<:ParamVolumeBounds,
    log_value::T   
end

export ConstantDensity


function ConstantDensity(bounds::ParamVolumeBounds, normalize::Bool)
    T = eltype(bounds)
    log_value = if normalize
        convert(T, - log_volume(bounds))
    else
        one(T)
    end
    ConstantDensity(bounds, log_value)
end


param_bounds(density::ConstantDensity) = density.bounds

nparams(density::ConstantDensity) = nparams(density.bounds)


function density_logval(
    density::ConstantDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    density.log_value
end


function density_logval!(
    r::AbstractArray{<:Real},
    density::ConstantDensity,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    size(params, 2) != length(r) && throw(ArgumentError("Number of parameter vectors doesn't match length of result"))
    fill!(r, density.log_value)
end
