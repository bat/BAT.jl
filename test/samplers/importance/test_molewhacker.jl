# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, Test, Distributions, LinearAlgebra, Statistics, StableRNGs
using DensityInterface: logdensityof
using MeasureBase: Likelihood, weightedmeasure
using ValueShapes: NamedTupleDist
import ForwardDiff, Optim

@testset "Molewhacker importance sampling" begin
    context(seed = 71; precision = Float64) = BATContext(rng = StableRNG(seed), ad = ForwardDiff, precision = precision)
    prior = MvNormal([0.0], [1.0;;])
    target = PosteriorMeasure(Likelihood(z -> Normal(z[1], 0.5), 2.0), prior)

    @testset "Gaussian estimator and adaptation" begin
        original = EvaluatedMeasure(target)
        alg = MolewhackerSampling(nsamples = 4000, batchsize = 512, maxiter = 5)
        em = evalmeasure(original, alg, context())
        smpls = BAT.samplesof(em)
        z = BAT.samplesof(em.empirical.transformed)
        q = Distribution(em.approx.transformed)
        info = em.evalinfo.result
        @test BAT.unevaluated(em) === BAT.unevaluated(original)
        @test length(smpls) == alg.nsamples
        @test mean(smpls)[1] ≈ 1.6 atol = 0.035
        @test var(smpls)[1] ≈ 0.2 atol = 0.025
        @test info.efficiency > 0.7
        @test probs(q)[1] >= alg.exploration_mass
        ratios = z.logd .- logpdf.(Ref(q), z.v)
        @test z.weight ≈ exp.(ratios .- info.logweight_scale)
        mass_estimate = mean(z.weight) * exp(info.logweight_scale)
        @test mass_estimate ≈ pdf(Normal(0, sqrt(1.25)), 2) rtol = 0.05
        @test BAT.validate_evalmeasure(em) === em
    end

    @testset "Fisher covariance and independent factors" begin
        c = atanh(0.5)
        model = z -> MvNormal([2z[1], 2z[1]], [4.0 2tanh(z[1]); 2tanh(z[1]) 1.0])
        p = PosteriorMeasure(Likelihood(model, [0.0, 0.0]), prior)
        seeds = [[c]]
        alg = MolewhackerSampling(nsamples = 32, maxiter = 0, nseeds = 1, init = ExplicitInit(seeds))
        em = evalmeasure(p, alg, context())
        q = Distribution(em.approx.transformed)
        @test cov(last(q.components))[1, 1] ≈ 4 / 25

        # Likelihood information at zero is 1 + 1 + 1. Add the prior once.
        product_model = z -> NamedTupleDist(a = Normal(z[1], 1.0), rest = NamedTupleDist(b = Poisson(exp(z[1])), c = Exponential(exp(z[1]))))
        p_product = PosteriorMeasure(Likelihood(product_model, (a = 0.0, rest = (b = 1, c = 1.0))), prior)
        em_product = evalmeasure(p_product, MolewhackerSampling(nsamples = 32, maxiter = 0,
            nseeds = 1, init = ExplicitInit([[0.0]])), context())
        @test cov(last(Distribution(em_product.approx.transformed).components))[1, 1] ≈ 1 / 4

        a, b = [-0.5], [1.5]
        repeated = [a, a, b]
        saved = deepcopy(repeated)
        merged = evalmeasure(target, MolewhackerSampling(nsamples = 32, maxiter = 0,
            nseeds = 3, init = ExplicitInit(repeated)), context())
        q_merged = Distribution(merged.approx.transformed)
        points = [[-0.5], [0.0], [1.5], [3.0]]
        expected = [0.1 * pdf(Normal(), x[1]) + 0.9 * (
            (2 / 3) * pdf(Normal(-0.5, sqrt(1 / 5)), x[1]) +
            (1 / 3) * pdf(Normal(1.5, sqrt(1 / 5)), x[1])) for x in points]
        @test pdf.(Ref(q_merged), points) ≈ expected
        @test merged.evalinfo.result.ncomponents == 3
        @test merged.evalinfo.result.ngeometries == 2
        @test repeated == saved
    end

    @testset "Optional center refinement in eight dimensions" begin
        p = PosteriorMeasure(Likelihood(z -> MvNormal(z, Matrix(0.25I, 8, 8)), fill(0.7, 8)),
            MvNormal(zeros(8), Matrix(1.0I, 8, 8)))
        alg = MolewhackerSampling(nsamples = 4000, batchsize = 250, maxiter = 4,
            maxevals = 6200, refine_centers = true)
        em = evalmeasure(p, alg, context())
        s = BAT.samplesof(em)
        @test maximum(abs, mean(s) .- 0.56) < 0.06
        @test maximum(abs, var(s) .- 0.2) < 0.04
        @test em.evalinfo.result.efficiency > 0.7
        @test em.evalinfo.result.nevals <= alg.maxevals
        @test length(s) == alg.nsamples
        q = Distribution(em.approx.transformed)
        z = BAT.samplesof(em.empirical.transformed)
        @test z.weight ≈ exp.(z.logd .- logpdf.(Ref(q), z.v) .- em.evalinfo.result.logweight_scale)
    end

    @testset "Separated nonlinear modes" begin
        p = PosteriorMeasure(Likelihood(z -> Normal(z[1]^2, 0.3), 4.0), prior)
        alg = MolewhackerSampling(nsamples = 6000, batchsize = 512, maxiter = 5,
            nseeds = 2, init = ExplicitInit([[-2.0], [2.0]]))
        em = evalmeasure(p, alg, context(72))
        s = BAT.samplesof(em)
        right_mass = sum(s.weight .* (first.(s.v) .> 0)) / sum(s.weight)
        @test right_mass ≈ 0.5 atol = 0.04
        @test mean(s)[1] ≈ 0 atol = 0.15
        @test em.evalinfo.result.efficiency > 0.5
    end

    @testset "Structured prior and coordinate law" begin
        p = PosteriorMeasure(Likelihood(x -> Normal(log(x.rate), 0.7), 0.4), NamedTupleDist(rate = LogNormal()))
        em = evalmeasure(p, MolewhackerSampling(nsamples = 3000, batchsize = 512, maxiter = 4), context(73))
        s, z = BAT.samplesof(em), BAT.samplesof(em.empirical.transformed)
        # log(rate) has an exact Gaussian posterior.
        μ, σ² = 0.4 / 1.49, 0.49 / 1.49
        logs = log.(getproperty.(s.v, :rate))
        @test sum(s.weight .* logs) / sum(s.weight) ≈ μ atol = 0.04
        @test sum(s.weight .* (logs .- μ).^2) / sum(s.weight) ≈ σ² atol = 0.035
        @test em.f_transform.(s.v) ≈ z.v
        @test s.weight == z.weight
        @test s.logd ≈ logdensityof.(Ref(BAT.unevaluated(em)), s.v)
        @test BAT.validate_evalmeasure(em) === em
    end

    @testset "RNG, precision, and target scale" begin
        alg = MolewhackerSampling(nsamples = 256, batchsize = 128, maxiter = 2, executor = BAT.SequentialExec())
        threaded = MolewhackerSampling(nsamples = 256, batchsize = 128, maxiter = 2, executor = BAT.MultiThreadedExec())
        a = evalmeasure(target, alg, context(74))
        b = evalmeasure(target, threaded, context(74))
        @test BAT.samplesof(a) == BAT.samplesof(b)
        shifted = evalmeasure(weightedmeasure(1000.0, BAT.batmeasure(target)), alg, context(74))
        @test BAT.samplesof(a).v ≈ BAT.samplesof(shifted).v rtol = 1e-6
        @test BAT.samplesof(a).weight ≈ BAT.samplesof(shifted).weight rtol = 1e-6

        fixed = MolewhackerSampling(nsamples = 256, maxiter = 0)
        a32 = evalmeasure(target, fixed, context(75, precision = Float32))
        b32 = evalmeasure(weightedmeasure(1e8, BAT.batmeasure(target)), fixed, context(75, precision = Float32))
        @test BAT.samplesof(a32).v == BAT.samplesof(b32).v
        @test BAT.samplesof(a32).weight ≈ BAT.samplesof(b32).weight rtol = 1e-6
        @test BAT.samplesof(a32).logd ≈ logdensityof.(Ref(BAT.unevaluated(a32)), BAT.samplesof(a32).v)
    end

    @testset "Budgets, pilot sizing, and mode initialization" begin
        fixed = evalmeasure(target, MolewhackerSampling(nsamples = 64, maxiter = 0, maxevals = 64), context())
        @test fixed.evalinfo.result.nevals == 64
        @test fixed.evalinfo.result.niterations == 0
        bounded = evalmeasure(target, MolewhackerSampling(nsamples = 64, batchsize = 64,
            maxiter = 10, maxevals = 320), context())
        @test bounded.evalinfo.result.nevals <= 320
        @test bounded.evalinfo.result.stop_reason == :maxevals

        flat = PosteriorMeasure(Likelihood(z -> Normal(0.0, 1.0), 0.0), prior)
        sized = evalmeasure(flat, MolewhackerSampling(nsamples = 500, target_ess = 50,
            batchsize = 64, maxiter = 0, maxevals = 564), context())
        @test 50 <= length(BAT.samplesof(sized)) <= 51
        @test sized.evalinfo.result.nevals == 64 + length(BAT.samplesof(sized))
        @test sized.evalinfo.result.ess ≈ length(BAT.samplesof(sized))
        plateau = evalmeasure(flat, MolewhackerSampling(nsamples = 64, batchsize = 64,
            maxiter = 3, patience = 1), context())
        @test plateau.evalinfo.result.stop_reason == :no_improvement
        @test plateau.evalinfo.result.ncomponents == 1

        seeded = evalmeasure(target, MolewhackerSampling(nsamples = 64, maxiter = 0, nseeds = 1,
            init = ExplicitInit([[0.0]]), mode = OptimAlg(optalg = Optim.LBFGS())), context())
        @test mean(last(Distribution(seeded.approx.transformed).components))[1] ≈ 1.6 atol = 1e-6
        limited = evalmeasure(target, MolewhackerSampling(nsamples = 64, maxiter = 0, nseeds = 1,
            init = ExplicitInit([[0.0]]), mode = OptimAlg(optalg = Optim.LBFGS()), maxevals = 65), context())
        @test limited.evalinfo.result.nevals == 65
        @test limited.evalinfo.result.stop_reason == :maxevals
        @test limited.evalinfo.result.ncomponents == 1
    end

    @testset "Defensive nonlinear tails" begin
        p = PosteriorMeasure(Likelihood(z -> Normal(2tanh(z[1]), 1.0), 0.0), prior)
        ε = 0.1
        em = evalmeasure(p, MolewhackerSampling(nsamples = 16, maxiter = 0,
            nseeds = 1, init = ExplicitInit([[0.0]]), exploration_mass = ε), context())
        q = Distribution(em.approx.transformed)
        points = [[0.0], [-12.0], [12.0]]
        # Fisher variance at zero is 1/5. This local Gaussian alone has infinite IS variance.
        expected = [log(ε * pdf(Normal(), x[1]) + (1 - ε) * pdf(Normal(0, sqrt(1 / 5)), x[1])) for x in points]
        @test logpdf.(Ref(q), points) ≈ expected
        logw = logdensityof.(Ref(BAT.batmeasure(p)), points) .- logpdf.(Ref(q), points)
        @test all(logw .<= logpdf(Normal(), 0.0) - log(ε))
    end

    @testset "Local failure and zero weights" begin
        # A singular seed cannot supply Fisher geometry. Keep the prior proposal.
        p = PosteriorMeasure(Likelihood(z -> Poisson(z[1] > 0 ? exp(z[1]) : 0.0), 1), prior)
        em = evalmeasure(p, MolewhackerSampling(nsamples = 256, maxiter = 0,
            nseeds = 1, init = ExplicitInit([[-1.0]])), context())
        s = BAT.samplesof(em)
        @test em.evalinfo.result.nfailed == 1
        @test em.evalinfo.result.ncomponents == 1
        @test all(iszero, s.weight[first.(s.v) .<= 0])
        @test sum(s.weight) > 0
        recovered = evalmeasure(p, MolewhackerSampling(nsamples = 32, maxiter = 0,
            nseeds = 3, init = ExplicitInit([[-1.0], [-1.0], [1.0]])), context())
        q_recovered = Distribution(recovered.approx.transformed)
        points = [[0.0], [1.0], [3.0]]
        expected = [0.1 * pdf(Normal(), x[1]) + 0.9 * pdf(Normal(1, sqrt(1 / (1 + exp(1)))), x[1]) for x in points]
        @test pdf.(Ref(q_recovered), points) ≈ expected
        @test recovered.evalinfo.result.ngeometries == 2
        @test recovered.evalinfo.result.nfailed == 1
        half_target = PosteriorMeasure(Likelihood(z -> Poisson(z[1] > 0 ? 1.0 : 0.0), 1), prior)
        no_mass_round = evalmeasure(half_target, MolewhackerSampling(nsamples = 128, batchsize = 1,
            maxiter = 5, patience = 1), BATContext(rng = BAT.Random.Xoshiro(6), ad = ForwardDiff))
        info = no_mass_round.evalinfo.result
        @test info.nevals == 129
        @test info.stop_reason == :no_improvement
        @test only(info.history) == (iteration = 1, accepted = false, gain = 0.0, ncomponents = 1)
        empty_target = PosteriorMeasure(Likelihood(z -> Normal(z[1], 1.0), Inf), prior)
        @test_throws ArgumentError evalmeasure(empty_target, MolewhackerSampling(nsamples = 16, maxiter = 0), context())
    end
end
