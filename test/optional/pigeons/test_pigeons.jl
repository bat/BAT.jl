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
    algorithm = PigeonsSampling(n_rounds = 1, n_chains = 3)
    context = BATContext(rng = Xoshiro(0x504947454f4e53))
    em = evalmeasure(posterior, algorithm, context)

    diagnostics = BAT.evalinfo(em).result
    @test diagnostics.adapted_betas ≈ [0, 0.5, 1]
    @test diagnostics.swap_acceptance_pr ≈ ones(2)
    @test diagnostics.global_barrier ≈ 0 atol = 100eps()
    @test Float64(massof(em)) ≈ 7.5 rtol = eps(Float32)
    @test BAT.evalinfo(em).result.lognormalizer ≈ log(2.5) rtol = 100eps()
    @test_throws ArgumentError evalmeasure(posterior, PigeonsSampling(n_rounds = 1, n_chains = 1))
    @test_throws ArgumentError evalmeasure(posterior, PigeonsSampling(n_rounds = 0, n_chains = 2))
end

@testset "shape and threaded replay" begin
    prior = NamedTupleDist(a = Uniform(-2.0, 4.0), b = LogNormal(0.2, 0.4))
    likelihood = logfuncdensity(x -> 0 <= x.a <= 2 ? -0.5 * (x.a - log(x.b))^2 : -Inf)
    posterior = PosteriorMeasure(likelihood, prior)
    initial_values = [(a = a, b = 1.0) for a in (0.0, 0.5, 1.0, 1.5)]
    sample(multithreaded) = evalmeasure(
        posterior,
        PigeonsSampling(
            n_rounds = 4, n_chains = 4, init = ExplicitInit(initial_values),
            multithreaded = multithreaded,
        ),
        BATContext(rng = Xoshiro(0x534841504544)),
    )
    em = sample(false)
    threaded_em = sample(true)
    smpls = samplesof(em)

    diagnostics = BAT.evalinfo(em).result
    @test all(x -> 0 <= x.a <= 2, smpls.v)
    @test diagnostics.global_barrier ≈ sum(1 .- diagnostics.swap_acceptance_pr)
    @test smpls == samplesof(threaded_em)
    @test massof(em) == massof(threaded_em)
    @test BAT.evalinfo(em).result == BAT.evalinfo(threaded_em).result
    @test length(smpls) == 2^4
    @test BAT.validate_evalmeasure(em) === em
    @test logdensityof(posterior).(smpls.v) ≈ smpls.logd

    from_empirical = evalmeasure(
        em, PigeonsSampling(n_rounds = 2, n_chains = 4),
        BATContext(rng = Xoshiro(0x454d50524943)),
    )
    @test all(x -> 0 <= x.a <= 2, samplesof(from_empirical).v)
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

@testset "variational reference" begin
    posterior = PosteriorMeasure(logfuncdensity(_ -> log(2.5)), 3.0 * batmeasure(Normal()))
    reference = Pigeons.GaussianReference(first_tuning_round = 2)
    reference_state = deepcopy((reference.mean, reference.standard_deviation))
    algorithm = PigeonsSampling(n_rounds = 2, n_chains = 4, variational = reference)
    em = evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(0x564152)))

    @test Float64(massof(em)) ≈ 7.5 rtol = eps(Float32)

    algorithm = PigeonsSampling(
        n_rounds = 6, n_chains = 4, n_chains_variational = 3, variational = reference,
    )
    sample() = evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(0x53544142)))
    em = sample()
    replay = sample()
    diagnostics = BAT.evalinfo(em).result

    @test Float64(massof(em)) ≈ 7.5 rtol = 0.1
    @test length(samplesof(em)) == 2^6
    @test samplesof(em) == samplesof(replay)
    @test massof(em) == massof(replay)
    @test (reference.mean, reference.standard_deviation) == reference_state
    @test diagnostics.global_barrier ≈ sum(1 .- diagnostics.swap_acceptance_pr)
    @test diagnostics.variational.global_barrier ≈ sum(1 .- diagnostics.variational.swap_acceptance_pr)
    @test ismissing(diagnostics.n_round_trips) && ismissing(diagnostics.n_tempered_restarts)
end
