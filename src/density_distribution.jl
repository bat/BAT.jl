# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MvDistDensity{D<:Distribution{Multivariate,Continuous}} <: AbstractDensity
    d::D
end

export MvDistDensity


function unsafe_density_logval(
    density::MvDistDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    Distributions.logpdf(density.d, params)
end

parent(density::MvDistDensity) = density.d

nparams(density::MvDistDensity) = length(density.d)

# Assume that implementations of logpdf are thread-safe and remote-safe:
exec_capabilities(::typeof(unsafe_density_logval), density::MvDistDensity, params::AbstractVector{<:Real}) =
    ExecCapabilities(0, true, 0, true)


function unsafe_density_logval!(
    r::AbstractArray{<:Real},
    density::MvDistDensity,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext = ExecContext()
)
    # TODO: Parallel execution, depending on exec_context
    Distributions.logpdf!(r, density.d, params)
end


# Assume that implementations of logpdf! are thread-safe and remote-safe:
exec_capabilities(::typeof(unsafe_density_logval!), r::AbstractArray{<:Real}, density::MvDistDensity, params::AbstractMatrix{<:Real}) =
    ExecCapabilities(0, true, 0, true) # Change when implementation of density_logval! for MvDistDensity becomes multithreaded.
