# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MvDistDensity{D<:Distribution{Multivariate,Continuous}} <: AbstractDensity
    d::D
end

export MvDistDensity

Base.convert(::Type{AbstractDensity}, d::Distribution{Multivariate,Continuous}) =
    MvDistDensity(d)


function density_logval(
    density::MvDistDensity,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    Distributions.logpdf(density.d, params)
end

Base.parent(density::MvDistDensity) = density.d

nparams(density::MvDistDensity) = length(density.d)

# Assume that implementations of logpdf are thread-safe and remote-safe:
exec_capabilities(::typeof(density_logval), density::MvDistDensity, params::AbstractVector{<:Real}) =
    ExecCapabilities(0, true, 0, true)


Distributions.sampler(density::MvDistDensity) = bat_sampler(parent(density))


Statistics.cov(density::MvDistDensity) = cov(density.d)
