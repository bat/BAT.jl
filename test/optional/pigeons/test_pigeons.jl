# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using DensityInterface
using Distributions
using MeasureBase: massof
using Random
using Statistics
using Test
using ValueShapes: NamedTupleDist

import Pigeons

@testset "reference normalization" begin
    posterior = PosteriorMeasure(logfuncdensity(_ -> log(2.5)), 3.0 * batmeasure(Normal()))
    algorithm = PigeonsSampling(n_rounds = 1, n_chains = 2)
    context = BATContext(rng = Xoshiro(0x504947454f4e53))
    em = evalmeasure(posterior, algorithm, context)

    @test Float64(massof(em)) ≈ 7.5 rtol = eps(Float32)
    @test BAT.evalinfo(em).result.lognormalizer ≈ log(2.5) rtol = 100eps()
    @test_throws ArgumentError evalmeasure(posterior, PigeonsSampling(n_rounds = 1, n_chains = 1))
    @test_throws ArgumentError evalmeasure(posterior, PigeonsSampling(n_rounds = 0, n_chains = 2))
end

@testset "shape and threaded replay" begin
    prior = NamedTupleDist(a = Uniform(-2.0, 4.0), b = LogNormal(0.2, 0.4))
    posterior = PosteriorMeasure(logfuncdensity(x -> -0.5 * (x.a - log(x.b))^2), prior)
    sample(multithreaded) = evalmeasure(
        posterior,
        PigeonsSampling(n_rounds = 4, n_chains = 4, multithreaded = multithreaded),
        BATContext(rng = Xoshiro(0x534841504544)),
    )
    em = sample(false)
    threaded_em = sample(true)
    smpls = samplesof(em)

    @test smpls == samplesof(threaded_em)
    @test massof(em) == massof(threaded_em)
    @test BAT.evalinfo(em).result == BAT.evalinfo(threaded_em).result
    @test length(smpls) == 2^4
    @test BAT.validate_evalmeasure(em) === em
    @test logdensityof(posterior).(smpls.v) ≈ smpls.logd
end

@testset "multimodal posterior" begin
    target_dist = MixtureModel([Normal(-4.0, 0.5), Normal(4.0, 0.5)])
    prior = Uniform(-8.0, 8.0)
    likelihood = logfuncdensity(x -> logpdf(target_dist, x) - logpdf(prior, x))
    posterior = PosteriorMeasure(likelihood, prior)
    algorithm = PigeonsSampling(n_rounds = 9, n_chains = 10)
    smpls = samplesof(
        evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(0x4d4f444553))),
    )

    @test 0.2 < count(<(-3), smpls.v) / length(smpls) < 0.8
    @test 0.2 < count(>(3), smpls.v) / length(smpls) < 0.8
end

@testset "gradient explorer" begin
    prior = Normal(-0.5, 1.2)
    observation, sigma = 1.25, 0.5
    likelihood = logfuncdensity(x -> logpdf(Normal(x, sigma), observation))
    posterior = PosteriorMeasure(likelihood, prior)
    algorithm = PigeonsSampling(
        n_rounds = 8,
        n_chains = 6,
        explorer = Pigeons.AutoMALA(base_n_refresh = 1),
    )
    em = evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(0x4155544f4d414c41)))
    posterior_var = inv(inv(var(prior)) + inv(sigma^2))
    posterior_mean = posterior_var * (mean(prior) / var(prior) + observation / sigma^2)
    evidence = pdf(Normal(mean(prior), sqrt(var(prior) + sigma^2)), observation)

    @test mean(samplesof(em).v) ≈ posterior_mean atol = 0.15
    @test Float64(massof(em)) ≈ evidence rtol = 0.25
end
