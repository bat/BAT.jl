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

@testset "constant likelihood" begin
    posterior = PosteriorMeasure(logfuncdensity(_ -> log(2.5)), 3.0 * batmeasure(Normal()))
    algorithm = PigeonsSampling(n_rounds = 4, n_chains = 4)
    context = BATContext(rng = Xoshiro(0x504947454f4e53))

    em = evalmeasure(posterior, algorithm, context)
    smpls = samplesof(em)

    @test BAT.validate_evalmeasure(em) === em
    @test logdensityof(posterior).(smpls.v) ≈ smpls.logd
    @test all(isone, smpls.weight)
    @test Float64(massof(em)) ≈ 7.5 rtol = eps(Float32)
    @test 0 < BAT.getess(BAT.empiricalof(em)) <= length(smpls)

    diagnostics = BAT.evalinfo(em).result
    @test diagnostics.lognormalizer ≈ log(2.5) rtol = 100eps()
    @test all(isfinite, diagnostics.lognormalizer_pair)
    @test diagnostics.n_tempered_restarts >= diagnostics.n_round_trips >= 0
end

@testset "shape and seed" begin
    prior =
        NamedTupleDist(a = Normal(1.5, 0.75), b = MvNormal([-1.0, 2.0], [1.0 0.2; 0.2 0.5]))
    posterior = PosteriorMeasure(logfuncdensity(_ -> 0.0), prior)
    algorithm =
        PigeonsSampling(n_rounds = 8, n_chains = 6, explorer = Pigeons.SliceSampler())
    sample(seed) =
        samplesof(evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(seed))))

    smpls = sample(0x534841504544)
    repeated_smpls = sample(0x534841504544)

    @test smpls == repeated_smpls
    @test mean(v.a for v in smpls.v) ≈ 1.5 atol = 0.25
    @test mean(v.b[1] for v in smpls.v) ≈ -1.0 atol = 0.3
    @test mean(v.b[2] for v in smpls.v) ≈ 2.0 atol = 0.25
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
    negative_fraction = count(<(0), smpls.v) / length(smpls)

    @test 0.2 < negative_fraction < 0.8
    @test minimum(smpls.v) < -3
    @test maximum(smpls.v) > 3
end

@testset "gradient explorer" begin
    posterior = PosteriorMeasure(logfuncdensity(_ -> 0.0), Normal())
    algorithm = PigeonsSampling(
        n_rounds = 2,
        n_chains = 3,
        explorer = Pigeons.AutoMALA(base_n_refresh = 1),
    )
    em = evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(0x4155544f4d414c41)))

    @test !isempty(samplesof(em))
    @test all(isfinite, samplesof(em).logd)
end

@testset "posterior required" begin
    @test_throws ArgumentError evalmeasure(
        Normal(),
        PigeonsSampling(n_rounds = 1, n_chains = 2),
        BATContext(rng = Xoshiro(0x504f5354)),
    )
end
