# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DistributionDensity{D<:Distribution{Multivariate,Continuous}} <: AbstractPriorDensity
    d::D
end

export DistributionDensity

Base.convert(::Type{AbstractDensity}, d::Distribution{Multivariate,Continuous}) =
    DistributionDensity(d)
    
Base.convert(::Type{AbstractPriorDensity}, d::Distribution{Multivariate,Continuous}) =
    DistributionDensity(d)


Base.parent(density::DistributionDensity) = density.d


function density_logval(
    density::DistributionDensity,
    params::Union{NamedTuple, AbstractVector{<:Real}}
)
    Distributions.logpdf(density.d, params)
end

param_bounds(density::DistributionDensity) = NoParamBounds(length(density.d))

params_shape(density::DistributionDensity) = valshape(density.d)

Distributions.sampler(density::DistributionDensity) = bat_sampler(parent(density))

Random.Sampler(rng::AbstractRNG, density::DistributionDensity, repetition::Val{1}) = sampler(density)

Statistics.cov(density::DistributionDensity) = cov(density.d)


param_bounds(density::DistributionDensity{<:NamedPrior}) = param_bounds(density.d)

