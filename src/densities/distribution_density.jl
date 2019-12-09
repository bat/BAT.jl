# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DistributionDensity{D<:Distribution{Multivariate,Continuous}} <: DistLikeDensity
    d::D
end


Base.convert(::Type{AbstractDensity}, d::Distribution{Multivariate,Continuous}) =
    DistributionDensity(d)
    
Base.convert(::Type{DistLikeDensity}, d::Distribution{Multivariate,Continuous}) =
    DistributionDensity(d)


Base.parent(density::DistributionDensity) = density.d


function density_logval(
    density::DistributionDensity,
    params::Any
)
    Distributions.logpdf(density.d, params)
end

param_bounds(density::DistributionDensity) = NoParamBounds(length(density.d))

params_shape(density::DistributionDensity) = varshape(density.d)

Distributions.sampler(density::DistributionDensity) = bat_sampler(parent(density))

Random.Sampler(rng::AbstractRNG, density::DistributionDensity, repetition::Val{1}) = sampler(density)

Statistics.cov(density::DistributionDensity) = cov(density.d)


param_bounds(density::DistributionDensity{<:NamedTupleDist}) = param_bounds(density.d)

