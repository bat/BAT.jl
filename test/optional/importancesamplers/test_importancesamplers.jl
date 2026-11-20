# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Test, BAT, Distributions, Random, Statistics, ValueShapes, LinearAlgebra
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

@testset "Target values without reevaluation" begin
    calls = Ref(0)
    likelihood = logfuncdensity(x -> (calls[] += 1; -abs(x[1])))
    target = PosteriorMeasure(likelihood, MvNormal(zeros(2), [1.0 0.0; 0.0 1.0]))
    sampler = IS.ImportanceSampling(IS.SphericalGaussian(zeros(2), 1.5); nsamples = 16)
    algorithm = ImportanceSamplersSampling(;
        sampler,
        pretransform = DoNotTransform(),
        threaded = false,
    )
    evaluated = evalmeasure(target, algorithm, BATContext(rng = Xoshiro(46)))
    @test calls[] == 16
    @test evalinfo(evaluated).result.samples.logtargets ≈ samplesof(evaluated).logd
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

@testset "Mapped parameter precision and composed-map fallback" begin
    prior = NamedTupleDist(rate = Gamma(2.0f0, 0.5f0), fraction = Beta(2.0f0, 3.0f0))
    target = PosteriorMeasure(logfuncdensity(x -> -abs2(x.rate)-abs2(x.fraction)), prior)
    sampler =
        IS.ImportanceSampling(IS.SphericalGaussian(zeros(Float32, 2), 1.0f0); nsamples = 64)
    evaluated = evalmeasure(
        target,
        ImportanceSamplersSampling(; sampler),
        BATContext(rng = Xoshiro(47)),
    )
    native = evalinfo(evaluated).result.samples
    transformed, f =
        BAT.transform_and_unshape(NormalBased(), BAT.EvaluatedMeasure(target), BATContext())
    expected =
        [BAT.with_logabsdet_jacobian(BAT.inverse(f), z) for z in eachcol(native.samples)]
    samples = samplesof(evaluated)
    @test (samples.v, typeof(first(samples.v))) ==
          (first.(expected), typeof(first(expected)[1]))
    @test (native.retained.logjacobians, eltype(native.retained.logjacobians)) ==
          (last.(expected), Float32)
    @test samples.logd ≈ logdensityof.(Ref(target), samples.v) rtol=32eps(Float32)

    scalar = PosteriorMeasure(logfuncdensity(x -> -abs2(x)), Beta(2, 3))
    sampler = IS.ImportanceSampling(IS.SphericalGaussian([0.0], 1.0); nsamples = 64)
    evaluated = evalmeasure(
        scalar,
        ImportanceSamplersSampling(; sampler),
        BATContext(rng = Xoshiro(48)),
    )
    samples = samplesof(evaluated)
    @test length(samples) == 64 && all(x -> 0 < x < 1, samples.v)
    @test samples.logd ≈ logdensityof.(Ref(scalar), samples.v)

    prior = NamedTupleDist(
        mat = reshape(MvNormal(zeros(4), Matrix{Float64}(I, 4, 4)), 2, 2),
        rate = Gamma(2, 0.5),
    )
    matrix_target = PosteriorMeasure(logfuncdensity(x -> -sum(abs2, x.mat)-x.rate), prior)
    sampler = IS.ImportanceSampling(IS.SphericalGaussian(zeros(5), 1.0); nsamples = 16)
    evaluated = evalmeasure(
        matrix_target,
        ImportanceSamplersSampling(; sampler),
        BATContext(rng = Xoshiro(49)),
    )
    samples = samplesof(evaluated)
    @test all(x -> size(x.mat) == (2, 2), samples.v)
    @test samples.logd ≈ logdensityof.(Ref(matrix_target), samples.v)
end

if Base.find_package("CUDA") !== nothing
    include("test_cuda.jl")
end
