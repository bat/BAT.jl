# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MvDistTargetDensity{D<:Distribution{Multivariate,Continuous}} <: AbstractTargetDensity
    d::D
end

export MvDistTargetDensity


function target_logval(
    target::MvDistTargetDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    Distributions.logpdf(target.d, params)
end

# Assume that implementations of logpdf are thread-safe and remote-safe:
exec_capabilities(::typeof(target_logval), target::MvDistTargetDensity, params::AbstractMatrix{<:Real}) =
    ExecCapabilities(0, true, 0, true)


function target_logval!(
    r::AbstractArray{<:Real},
    target::MvDistTargetDensity,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext = ExecContext()
)
    # TODO: Parallel execution, depending on exec_context
    Distributions.logpdf!(r, target.d, params)
end


# Assume that implementations of logpdf! are thread-safe and remote-safe:
exec_capabilities(::typeof(target_logval!), target::MvDistTargetDensity, params::AbstractMatrix{<:Real}) =
    ExecCapabilities(0, true, 0, true) # Change when implementation of target_logval! for MvDistTargetDensity becomes multithreaded.
