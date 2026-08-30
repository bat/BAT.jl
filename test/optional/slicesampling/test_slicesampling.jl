# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using DensityInterface
using Distributions
using Random
using Statistics
using Test
using ValueShapes

import SliceSampling

@testset "conjugate posterior" begin
    prior = Normal(-0.5, 1.2)
    observation, observation_std = 1.25, 0.5
    likelihood = logfuncdensity(x -> logpdf(Normal(x, observation_std), observation))
    posterior = PosteriorMeasure(likelihood, prior)
    algorithm = SliceMCMCSampling(nsamples = 4_000, n_burnin = 500)
    context = BATContext(rng = Xoshiro(0x534c494345))

    em = evalmeasure(posterior, algorithm, context)
    smpls = samplesof(em)
    posterior_var = inv(inv(var(prior)) + inv(observation_std^2))
    posterior_mean =
        posterior_var * (mean(prior) / var(prior) + observation / observation_std^2)

    @test BAT.validate_evalmeasure(em) === em
    @test length(smpls) == algorithm.nsamples
    @test logdensityof(posterior).(smpls.v) ≈ smpls.logd
    @test all(isone, smpls.weight)
    @test mean(smpls) ≈ posterior_mean atol = 0.04
    @test var(smpls) ≈ posterior_var rtol = 0.12
    @test 0 < BAT.getess(BAT.empiricalof(em)) <= length(smpls)
end

@testset "shape, sampler, and seed" begin
    prior = NamedTupleDist(a = Normal(1.5, 0.75), b = Normal(-2.0, 0.5))
    posterior = PosteriorMeasure(logfuncdensity(_ -> 0.0), prior)
    sampler = SliceSampling.RandPermGibbs(SliceSampling.SliceDoublingOut(1.0))
    algorithm = SliceMCMCSampling(sampler = sampler, nsamples = 256, n_burnin = 64)
    draw(seed) =
        samplesof(evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(seed))))

    smpls = draw(0x5348415045)

    @test smpls == draw(0x5348415045)
    @test first(smpls.v) isa NamedTuple{(:a, :b)}
    @test logdensityof(posterior).(smpls.v) ≈ smpls.logd
    @test BAT.evalinfo(evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(1)))).algorithm ===
          algorithm
end
