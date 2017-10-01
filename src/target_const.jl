# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ConstTargetDensity{T<:Real} <: AbstractTargetDensity
    log_value::T
end

export ConstTargetDensity

function target_logval(
    target::ConstTargetDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    target.log_value
end


function target_logval!(
    r::AbstractArray{<:Real},
    target::ConstTargetDensity,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext
)
    @assert size(params, 2) == length(r)
    fill!(r, target.log_value)
end
