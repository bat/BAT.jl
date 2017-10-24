# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ConstDensityFunction{T<:Real} <: AbstractDensityFunction
    log_value::T
    nparams::Int
end

export ConstDensityFunction

nparams(density::ConstDensityFunction) = density.nparams

function density_logval(
    density::ConstDensityFunction,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    density.log_value
end


function density_logval!(
    r::AbstractArray{<:Real},
    density::ConstDensityFunction,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    size(params, 2) != length(r) && throw(ArgumentError("Number of parameter vectors doesn't match length of result"))
    fill!(r, density.log_value)
end
