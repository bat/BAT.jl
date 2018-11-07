# This file is a part of BAT.jl, licensed under the MIT License (MIT).


export BATChisqSampler

struct BATChisqSampler{T} <: BATSampler{Univariate,Continuous}
    gamma_sampler::T
end

function BATChisqSampler(d::Chisq)
    shape = dof(d) / 2
    scale = typeof(shape)(2)
    BATChisqSampler(BATGammaMTSampler(shape, scale))
end

Base.eltype(s::BATChisqSampler) = eltype(s.gamma_sampler)

Random.rand(rng::AbstractRNG, s::BATChisqSampler) = rand(rng, s.gamma_sampler)

bat_sampler(d::Chisq) = BATChisqSampler(d)
