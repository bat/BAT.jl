# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MvDistDensity{D<:Distribution{Multivariate,Continuous}} <: AbstractDensity
    d::D
end

export MvDistDensity

Base.convert(::Type{AbstractDensity}, d::Distribution{Multivariate,Continuous}) =
    MvDistDensity(d)


function density_logval(
    density::MvDistDensity,
    params::AbstractVector{<:Real}
)
    Distributions.logpdf(density.d, params)
end

Base.parent(density::MvDistDensity) = density.d

nparams(density::MvDistDensity) = length(density.d)


Distributions.sampler(density::MvDistDensity) = bat_sampler(parent(density))


Statistics.cov(density::MvDistDensity) = cov(density.d)
