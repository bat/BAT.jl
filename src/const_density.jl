# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ConstDensity{T<:Real} <: UnconstrainedDensityFunction
    B<:ParamVolumeBounds,
    log_value::T   
end

export ConstDensity


# ToDo: XXXX !!!!
function ConstDensity(bounds::ParamVolumeBounds, normalize::Bool)
    log_value = if normalize
        1 / ...


param_bounds(density::ConstDensity) = density.bounds
nparams(density::ConstDensity) = nparams(density.bounds)

function density_logval(
    density::ConstDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    density.log_value
end


function density_logval!(
    r::AbstractArray{<:Real},
    density::ConstDensity,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    size(params, 2) != length(r) && throw(ArgumentError("Number of parameter vectors doesn't match length of result"))
    fill!(r, density.log_value)
end
