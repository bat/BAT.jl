# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MvDistDensityFunction{D<:Distribution{Multivariate,Continuous}} <: AbstractDensityFunction
    d::D
end

export MvDistDensityFunction


function density_logval(
    density::MvDistDensityFunction,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    Distributions.logpdf(density.d, params)
end

parent(density::MvDistDensityFunction) = density.d

nparams(density::MvDistDensityFunction) = length(density.d)

# Assume that implementations of logpdf are thread-safe and remote-safe:
exec_capabilities(::typeof(density_logval), density::MvDistDensityFunction, params::AbstractVector{<:Real}) =
    ExecCapabilities(0, true, 0, true)


function density_logval!(
    r::AbstractArray{<:Real},
    density::MvDistDensityFunction,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext = ExecContext()
)
    # TODO: Parallel execution, depending on exec_context
    Distributions.logpdf!(r, density.d, params)
end


# Assume that implementations of logpdf! are thread-safe and remote-safe:
exec_capabilities(::typeof(density_logval!), density::MvDistDensityFunction, params::AbstractMatrix{<:Real}) =
    ExecCapabilities(0, true, 0, true) # Change when implementation of density_logval! for MvDistDensityFunction becomes multithreaded.
