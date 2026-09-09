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
        alg = MolewhackerSampling(nsamples = 4000, batchsize = 512, maxiter = 5)
        em = evalmeasure(target, alg, context())
        smpls = BAT.samplesof(em)
        z = BAT.samplesof(em.empirical.transformed)
        q = Distribution(em.approx.transformed)
        info = em.evalinfo.result
        @test mean(smpls)[1] ≈ 1.6 atol = 0.035
        @test var(smpls)[1] ≈ 0.2 atol = 0.025
        @test info.efficiency > 0.7
        @test probs(q)[1] >= alg.exploration_mass
        ratios = z.logd .- logpdf.(Ref(q), z.v)
        @test z.weight ≈ exp.(ratios .- info.logweight_scale)
        mass_estimate = mean(z.weight) * exp(info.logweight_scale)
        @test mass_estimate ≈ pdf(Normal(0, sqrt(1.25)), 2) rtol = 0.05
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

        # Likelihood information at zero is 1/4 + 2 + 4. Add the prior once.
        product_model = z -> NamedTupleDist(a = Normal(z[1], 2.0), rest = NamedTupleDist(b = Poisson(2exp(z[1])), c = Exponential(3exp(2z[1]))))
        p_product = PosteriorMeasure(Likelihood(product_model, (a = 0.0, rest = (b = 2, c = 3.0))), prior)
        em_product = evalmeasure(p_product, MolewhackerSampling(nsamples = 32, maxiter = 0,
            nseeds = 1, init = ExplicitInit([[0.0]])), context())
        @test cov(last(Distribution(em_product.approx.transformed).components))[1, 1] ≈ 4 / 29

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
        @test repeated == saved
    end

    @testset "Optional center refinement in eight dimensions" begin
        p = PosteriorMeasure(Likelihood(z -> MvNormal(z, Matrix(0.25I, 8, 8)), fill(0.7, 8)),
            MvNormal(zeros(8), Matrix(1.0I, 8, 8)))
        alg = MolewhackerSampling(nsamples = 1000, batchsize = 250, maxiter = 4,
            refine_centers = true)
        em = evalmeasure(p, alg, context())
        s = BAT.samplesof(em)
        @test maximum(abs, mean(s) .- 0.56) < 0.06
        @test em.evalinfo.result.efficiency > 0.7
    end

    @testset "Separated nonlinear modes" begin
        p = PosteriorMeasure(Likelihood(z -> Normal(z[1]^2, 0.3), 4.0), prior)
        alg = MolewhackerSampling(nsamples = 6000, batchsize = 512, maxiter = 5,
            nseeds = 2, init = ExplicitInit([[-2.0], [2.0]]))
        em = evalmeasure(p, alg, context(72))
        s = BAT.samplesof(em)
        right_mass = sum(s.weight .* (first.(s.v) .> 0)) / sum(s.weight)
        @test right_mass ≈ 0.5 atol = 0.04
    end

    @testset "Structured prior and coordinate law" begin
        p = PosteriorMeasure(Likelihood(x -> Normal(log(x.rate), 0.7), 0.4), NamedTupleDist(rate = LogNormal()))
        em = evalmeasure(p, MolewhackerSampling(nsamples = 16, maxiter = 0,
            nseeds = 1, init = ExplicitInit([(rate = 1.0,)])), context(73))
        @test cov(last(Distribution(em.approx.transformed).components))[1, 1] ≈ 0.49 / 1.49
        # Validate both proposal and empirical coordinate pairs, including the Jacobian.
        @test BAT.validate_evalmeasure(em; context = context(73)) === em
    end

    @testset "RNG, precision, and target scale" begin
        alg = MolewhackerSampling(nsamples = 256, batchsize = 128, maxiter = 2, executor = BAT.SequentialExec())
        threaded = MolewhackerSampling(nsamples = 256, batchsize = 128, maxiter = 2, executor = BAT.MultiThreadedExec())
        a = evalmeasure(target, alg, context(74, precision = Float32))
        b = evalmeasure(target, threaded, context(74, precision = Float32))
        @test BAT.samplesof(a) == BAT.samplesof(b)
        shifted = evalmeasure(weightedmeasure(1e8, BAT.batmeasure(target)), alg, context(74, precision = Float32))
        @test BAT.samplesof(a).v ≈ BAT.samplesof(shifted).v rtol = 1e-6
        @test BAT.samplesof(a).weight ≈ BAT.samplesof(shifted).weight rtol = 1e-6
        @test BAT.samplesof(a).logd ≈ logdensityof.(Ref(BAT.unevaluated(a)), BAT.samplesof(a).v)
    end

    @testset "Budgets, pilot sizing, and mode initialization" begin
        bounded = evalmeasure(target, MolewhackerSampling(nsamples = 64, batchsize = 64,
            maxiter = 10, maxevals = 320, refine_centers = true), context())
        @test bounded.evalinfo.result.nevals <= 320
        @test length(BAT.samplesof(bounded)) == 64

        flat = PosteriorMeasure(Likelihood(z -> Normal(0.0, 1.0), 0.0), prior)
        sized = evalmeasure(flat, MolewhackerSampling(nsamples = 500, target_ess = 50,
            batchsize = 64, maxiter = 0, maxevals = 564), context())
        @test 50 <= length(BAT.samplesof(sized)) <= 51
        @test sized.evalinfo.result.nevals == 64 + length(BAT.samplesof(sized))

        seeded = evalmeasure(target, MolewhackerSampling(nsamples = 64, maxiter = 0, nseeds = 1,
            init = ExplicitInit([[0.0]]), mode = OptimAlg(optalg = Optim.LBFGS())), context())
        @test mean(last(Distribution(seeded.approx.transformed).components))[1] ≈ 1.6 atol = 1e-6
        limited = evalmeasure(target, MolewhackerSampling(nsamples = 64, maxiter = 0, nseeds = 1,
            init = ExplicitInit([[0.0]]), mode = OptimAlg(optalg = Optim.LBFGS()), maxevals = 65), context())
        @test limited.evalinfo.result.nevals == 65
    end

    @testset "Defensive nonlinear tails" begin
        p = PosteriorMeasure(Likelihood(z -> Normal(2tanh(z[1]), 1.0), 0.0), prior)
        ε = 0.1
        em = evalmeasure(p, MolewhackerSampling(nsamples = 16, maxiter = 0,
            nseeds = 1, init = ExplicitInit([[0.0]]), exploration_mass = ε), context())
        q = Distribution(em.approx.transformed)
        points = [[0.0], [-12.0], [12.0]]
        # Fisher variance at zero is 1/5. This local Gaussian alone has infinite IS variance.
        logw = logdensityof.(Ref(BAT.batmeasure(p)), points) .- logpdf.(Ref(q), points)
        @test all(logw .<= logpdf(Normal(), 0.0) - log(ε))
    end

    @testset "Local failure and zero weights" begin
        # A singular seed cannot supply Fisher geometry. Keep the prior proposal.
        p = PosteriorMeasure(Likelihood(z -> Poisson(z[1] > 0 ? exp(z[1]) : 0.0), 1), prior)
        em = evalmeasure(p, MolewhackerSampling(nsamples = 256, maxiter = 0,
            nseeds = 1, init = ExplicitInit([[-1.0]])), context())
        s = BAT.samplesof(em)
        @test logpdf.(Ref(Distribution(em.approx.transformed)), s.v) ≈ logpdf.(Ref(prior), s.v)
        @test all(iszero, s.weight[first.(s.v) .<= 0])
        @test sum(s.weight) > 0
        recovered = evalmeasure(p, MolewhackerSampling(nsamples = 32, maxiter = 0,
            nseeds = 3, init = ExplicitInit([[-1.0], [-1.0], [1.0]])), context())
        q_recovered = Distribution(recovered.approx.transformed)
        points = [[0.0], [1.0], [3.0]]
        expected = [0.1 * pdf(Normal(), x[1]) + 0.9 * pdf(Normal(1, sqrt(1 / (1 + exp(1)))), x[1]) for x in points]
        @test pdf.(Ref(q_recovered), points) ≈ expected
        half_target = PosteriorMeasure(Likelihood(z -> Poisson(z[1] > 0 ? 1.0 : 0.0), 1), prior)
        no_mass_round = evalmeasure(half_target, MolewhackerSampling(nsamples = 128, batchsize = 1,
            maxiter = 5, patience = 1), BATContext(rng = BAT.Random.Xoshiro(6), ad = ForwardDiff))
        info = no_mass_round.evalinfo.result
        @test info.nevals == 129
        @test info.stop_reason == :no_improvement
    end
end
