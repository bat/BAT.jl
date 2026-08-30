# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using DensityInterface
using Distributions
using Random
using Statistics
using Test
using ValueShapes

import EllipticalSliceSampling

@testset "conjugate posterior" begin
    prior = Normal(-0.5, 1.2)
    observation, observation_std = 1.25, 0.5
    likelihood = logfuncdensity(x -> logpdf(Normal(x, observation_std), observation))
    posterior = PosteriorMeasure(likelihood, prior)
    algorithm = EllipticalSliceMCMCSampling(nsamples = 2_000, n_burnin = 250)
    context = BATContext(rng = Xoshiro(0x454c4c49505345))

    em = evalmeasure(posterior, algorithm, context)
    smpls = samplesof(em)
    posterior_var = inv(inv(var(prior)) + inv(observation_std^2))
    posterior_mean =
        posterior_var * (mean(prior) / var(prior) + observation / observation_std^2)

    @test length(smpls) == algorithm.nsamples
    @test logdensityof(posterior).(smpls.v) ≈ smpls.logd
    @test mean(smpls) ≈ posterior_mean atol = 0.06
    @test var(smpls) ≈ posterior_var rtol = 0.18
end

@testset "transformed structured prior" begin
    prior = NamedTupleDist(a = Uniform(-2.0, 4.0), b = LogNormal(0.2, 0.4))
    posterior = PosteriorMeasure(logfuncdensity(_ -> 0.0), prior)
    algorithm = EllipticalSliceMCMCSampling(nsamples = 128, n_burnin = 32)
    smpls = samplesof(evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(0x5052494f52))))

    @test all(s -> -2 <= s.a <= 4 && s.b > 0, smpls.v)
    @test logdensityof(posterior).(smpls.v) ≈ smpls.logd
end
