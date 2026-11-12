# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test, BAT, Distributions, Random, Statistics, ValueShapes
import DensityInterface: logfuncdensity, logdensityof
import ForwardDiff
import ImportanceSamplers as IS

@testset "Native estimator mapping" begin
    target = MvNormal([0.25, -0.5], [1.0 0.4; 0.4 2.0])
    sampler = IS.ImportanceSampling(IS.SphericalGaussian(zeros(2), 1.5); nsamples = 64)
    algorithm = ImportanceSamplersSampling(;
        sampler,
        pretransform = DoNotTransform(),
        threaded = false,
    )
    raw =
        IS.importance_sample(Xoshiro(42), x -> logpdf(target, x), sampler; threaded = false)
    evaluated = evalmeasure(target, algorithm, BATContext(rng = Xoshiro(42)))
    samples = samplesof(evaluated)

    @test reduce(hcat, samples.v) == raw.samples
    @test evalinfo(evaluated).result.samples.logweights ≈ raw.logweights
    @test samples.weight ≈ IS.normalized_weights(raw)
    @test samples.logd ≈ logpdf.(Ref(target), samples.v)
    @test log(BAT.massof(evaluated)) ≈ IS.lognormalizer(raw)
    @test mean(samples) ≈ mean(raw)
end

@testset "Shapes, Jacobians and target scale" begin
    prior = NamedTupleDist(rate = Gamma(2, 0.5), fraction = Beta(2, 3))
    sampler = IS.ImportanceSampling(IS.SphericalGaussian(zeros(2), 1.0); nsamples = 64)
    algorithm = ImportanceSamplersSampling(; sampler, threaded = true)
    for shift in (0.0, 1000.0, -1000.0)
        target = PosteriorMeasure(logfuncdensity(_ -> shift), prior)
        initial = BAT.EvaluatedMeasure(target)
        evaluated = evalmeasure(initial, algorithm, BATContext(rng = Xoshiro(43)))
        samples = samplesof(evaluated)
        @test BAT.unevaluated(evaluated) === BAT.unevaluated(initial)
        @test samples.weight ≈ fill(1/64, 64) atol=2e-10
        @test samples.logd ≈ logdensityof.(Ref(target), samples.v) atol=2e-10
        @test log(BAT.massof(evaluated)) ≈ shift atol=2e-10
        @test BAT.validate_evalmeasure(
            evaluated;
            context = BATContext(rng = Xoshiro(45)),
        ) === evaluated
    end
end

@testset "Adaptive provenance and BAT AD" begin
    bank = IS.ProposalBank([
        IS.SphericalGaussian([-0.5, 0.0], 1.0),
        IS.SphericalGaussian([0.5, 0.0], 1.0),
    ])
    transition = IS.HamiltonianMonteCarlo(;
        step_size = 0.1,
        leapfrog_steps = 2,
        tuning = IS.WarmupTuning(0),
    )
    sampler = IS.HAIS(bank; rounds = 2, round_size = 32, transition)
    algorithm = ImportanceSamplersSampling(;
        sampler,
        pretransform = DoNotTransform(),
        threaded = true,
    )
    target = MvNormal(zeros(2), [1.0 0.3; 0.3 1.0])
    analytic = IS.LogTarget(x -> logpdf(target, x); grad = x -> -(cov(target) \ x))
    expected = IS.importance_sample(Xoshiro(44), analytic, sampler; threaded = true)
    evaluated =
        evalmeasure(target, algorithm, BATContext(rng = Xoshiro(44), ad = ForwardDiff))
    samples = samplesof(evaluated)
    raw = evalinfo(evaluated).result.samples
    @test length(samples) == 64
    @test samples.logd ≈ logpdf.(Ref(target), samples.v)
    @test samples.weight ≈ IS.normalized_weights(raw)
    @test all(
        key -> getproperty(samples.info, key) == getproperty(raw.provenance, key),
        keys(raw.provenance),
    )
    @test raw.samples ≈ expected.samples
    @test raw.logweights ≈ expected.logweights
end
