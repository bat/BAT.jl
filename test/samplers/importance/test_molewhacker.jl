# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, Test, Distributions, LinearAlgebra, Statistics, StableRNGs
using DensityInterface: logdensityof
using MeasureBase: Likelihood, weightedmeasure
using ValueShapes: NamedTupleDist
import ForwardDiff, Optim, OptimizationLBFGSB

@testset "Molewhacker importance sampling" begin
    context(seed = 71; precision = Float64) = BATContext(rng = StableRNG(seed), ad = ForwardDiff, precision = precision)
    prior = MvNormal([0.0], [1.0;;])
    target = PosteriorMeasure(Likelihood(z -> Normal(z[1], 0.5), 2.0), prior)

    @testset "Gaussian estimator and adaptation" begin
        alg = MolewhackerSampling(nsamples = 4000, batchsize = 512, maxiter = 5, nseeds = 0, exploration_mass = 0.1)
        em = evalmeasure(target, alg, context())
        smpls = BAT.samplesof(em)
        z = BAT.samplesof(em.empirical.transformed)
        q = Distribution(em.approx.transformed)
        info = em.evalinfo.result
        @test mean(smpls)[1] ≈ 1.6 atol = 0.035
        @test var(smpls)[1] ≈ 0.2 atol = 0.025
        @test info.efficiency > 0.7
        uniform = MixtureModel(q.components)
        masses = [pdf(Normal(1.6, sqrt(0.2)), mean(c)[1]) / pdf(uniform, mean(c)) for c in q.components]
        masses .*= (1 - alg.exploration_mass) / sum(masses)
        masses[1] += alg.exploration_mass
        @test probs(q) ≈ masses
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
        alg = MolewhackerSampling(nsamples = 32, maxiter = 0, nseeds = 1, init_mode = nothing, init = ExplicitInit(seeds))
        em = evalmeasure(p, alg, context())
        q = Distribution(em.approx.transformed)
        @test cov(last(q.components))[1, 1] ≈ 4 / 25

        # Compact Normal covariance charts retain the variance-derivative information.
        for (model, variance) in ((z -> MvNormal([2z[1], z[1]], Diagonal(exp.([2z[1], z[1]]))), 2 / 17),
                (z -> MvNormal([2z[1], z[1]], exp(2z[1])I(2)), 1 / 10))
            p_compact = PosteriorMeasure(Likelihood(model, [0.0, 0.0]), prior)
            em_compact = evalmeasure(p_compact, MolewhackerSampling(nsamples = 32, maxiter = 0,
                nseeds = 1, init_mode = nothing, init = ExplicitInit([[0.0]])), context())
            @test cov(last(Distribution(em_compact.approx.transformed).components))[1, 1] ≈ variance
        end

        # Likelihood information at zero is 1/4 + 2 + 4. Add the prior once.
        product_model = z -> NamedTupleDist(a = Normal(z[1], 2.0), rest = NamedTupleDist(b = Poisson(2exp(z[1])), c = Exponential(3exp(2z[1]))))
        p_product = PosteriorMeasure(Likelihood(product_model, (a = 0.0, rest = (b = 2, c = 3.0))), prior)
        em_product = evalmeasure(p_product, MolewhackerSampling(nsamples = 32, maxiter = 0,
            nseeds = 1, init_mode = nothing, init = ExplicitInit([[0.0]])), context())
        @test cov(last(Distribution(em_product.approx.transformed).components))[1, 1] ≈ 4 / 29

        a, b = [-0.5], [1.5]
        repeated = [a, a, b]
        saved = deepcopy(repeated)
        repeated_result = evalmeasure(target, MolewhackerSampling(nsamples = 32, maxiter = 0,
            nseeds = 3, init_mode = nothing, init = ExplicitInit(repeated)), context())
        q_repeated = Distribution(repeated_result.approx.transformed)
        points = [[-0.5], [0.0], [1.5], [3.0]]
        ga, gb = Normal(-0.5, sqrt(0.2)), Normal(1.5, sqrt(0.2))
        uniform(x) = (2pdf(ga, x) + pdf(gb, x)) / 3
        masses = [2pdf(Normal(1.6, sqrt(0.2)), -0.5) / uniform(-0.5),
            pdf(Normal(1.6, sqrt(0.2)), 1.5) / uniform(1.5)]
        masses ./= sum(masses)
        expected = [masses[1] * pdf(ga, x[1]) + masses[2] * pdf(gb, x[1]) for x in points]
        @test pdf.(Ref(q_repeated), points) ≈ expected
        @test repeated == saved
    end

    @testset "Separated nonlinear modes" begin
        p = PosteriorMeasure(Likelihood(z -> Normal(z[1]^2, 0.3), 4.0), prior)
        alg = MolewhackerSampling(nsamples = 6000, batchsize = 512, maxiter = 5,
            nseeds = 2, init_mode = nothing, init = ExplicitInit([[-2.0], [2.0]]))
        em = evalmeasure(p, alg, context(72))
        s = BAT.samplesof(em)
        right_mass = sum(s.weight .* (first.(s.v) .> 0)) / sum(s.weight)
        @test right_mass ≈ 0.5 atol = 0.04
    end

    @testset "Structured prior and coordinate law" begin
        p = PosteriorMeasure(Likelihood(x -> Normal(log(x.rate), 0.7), 0.4), NamedTupleDist(rate = LogNormal()))
        em = evalmeasure(p, MolewhackerSampling(nsamples = 16, maxiter = 0,
            nseeds = 1, init_mode = nothing, init = ExplicitInit([(rate = 1.0,)])), context(73))
        @test cov(last(Distribution(em.approx.transformed).components))[1, 1] ≈ 0.49 / 1.49
        # Validate both proposal and empirical coordinate pairs, including the Jacobian.
        @test BAT.validate_evalmeasure(em; context = context(73)) === em
    end

    @testset "RNG, precision, and target scale" begin
        alg = MolewhackerSampling(nsamples = 256, batchsize = 128, maxiter = 2, executor = BAT.SequentialExec())
        threaded = MolewhackerSampling(nsamples = 256, batchsize = 128, maxiter = 2, executor = BAT.MultiThreadedExec(ntasks = 2))
        a = evalmeasure(target, alg, context(74, precision = Float32))
        b = evalmeasure(target, threaded, context(74, precision = Float32))
        @test BAT.samplesof(a) == BAT.samplesof(b)
        # Test weight-scale invariance away from tied scores in the exact
        # Gaussian proposal produced by mode initialization.
        scale_alg = MolewhackerSampling(nsamples = 256, batchsize = 128, maxiter = 2, nseeds = 0)
        a = evalmeasure(target, scale_alg, context(74, precision = Float32))
        shifted = evalmeasure(weightedmeasure(1e8, BAT.batmeasure(target)), scale_alg, context(74, precision = Float32))
        @test BAT.samplesof(a).v ≈ BAT.samplesof(shifted).v rtol = 1e-6
        @test BAT.samplesof(a).weight ≈ BAT.samplesof(shifted).weight rtol = 1e-6
        @test BAT.samplesof(a).logd ≈ logdensityof.(Ref(BAT.unevaluated(a)), BAT.samplesof(a).v)
    end

    @testset "Bounded target concurrency" begin
        workers, guard = Set{Task}(), ReentrantLock()
        function model(z)
            lock(guard) do
                push!(workers, current_task())
            end
            return Normal(z[1], 0.5)
        end
        p = PosteriorMeasure(Likelihood(model, 2.0), prior)
        for (ntasks, expected) in ((1, 1), (2, 3))
            empty!(workers)
            alg = MolewhackerSampling(nsamples = 17, maxiter = 0, nseeds = 0,
                executor = BAT.MultiThreadedExec(ntasks = ntasks))
            evalmeasure(p, alg, context())
            # The first point uses the caller. A one-task limit keeps all work there.
            @test length(workers) == expected
        end
    end

    @testset "Budgets, pilot sizing, and mode initialization" begin
        bounded = evalmeasure(target, MolewhackerSampling(batchsize = 64,
            maxiter = 10, maxevals = 320, nseeds = 0), context())
        @test bounded.evalinfo.result.nevals <= 320
        @test length(BAT.samplesof(bounded)) == bounded.evalinfo.result.npilot

        flat = PosteriorMeasure(Likelihood(z -> Normal(0.0, 1.0), 0.0), prior)
        sized = evalmeasure(flat, MolewhackerSampling(target_ess = 12_000,
            batchsize = 64, maxiter = 0, maxevals = 20_064, nseeds = 0), context())
        @test 12_000 <= length(BAT.samplesof(sized)) <= 12_001
        @test sized.evalinfo.result.nevals == 64 + length(BAT.samplesof(sized))
        budgets = evalmeasure(flat, MolewhackerSampling(nsamples = 256, target_ess = 32,
            batchsize = 64, maxiter = 1, nseeds = 0), context())
        @test budgets.evalinfo.result.niterations == 1
        for (rule, reason) in (((; target_pool_ess = 32), :pilot_ess),
                ((; target_efficiency = 128 / 256), :pool_efficiency))
            stopped = evalmeasure(flat, MolewhackerSampling(; nsamples = 256, target_ess = 128,
                batchsize = 64, maxiter = 10, nseeds = 0, rule...), context())
            info = stopped.evalinfo.result
            @test (info.niterations, info.ncomponents, info.stop_reason) == (0, 1, reason)
            @test info.ess ≈ 128
        end

        concentrated = PosteriorMeasure(Likelihood(z -> MvNormal(z, 0.25I(18)), fill(2.0, 18)),
            MvNormal(zeros(18), I(18)))
        seeded = evalmeasure(concentrated, MolewhackerSampling(nsamples = 1000, maxiter = 0, maxcomponents = 10), context())
        @test seeded.evalinfo.result.efficiency > 0.8
        @test maximum(abs, mean(BAT.samplesof(seeded)) .- 1.6) < 0.06
        limited = evalmeasure(target, MolewhackerSampling(nsamples = 64, maxiter = 0, nseeds = 1,
            init = ExplicitInit([[0.0]]), init_mode = OptimAlg(optalg = Optim.LBFGS()), maxevals = 66), context())
        @test limited.evalinfo.result.nevals <= 66
        @test limited.evalinfo.result.nseed_exhausted == 1
    end

    @testset "Defensive nonlinear tails" begin
        p = PosteriorMeasure(Likelihood(z -> Normal(2tanh(z[1]), 1.0), 0.0), prior)
        ε = 0.1
        em = evalmeasure(p, MolewhackerSampling(nsamples = 16, maxiter = 0,
            nseeds = 1, init_mode = nothing, init = ExplicitInit([[0.0]]), exploration_mass = ε), context())
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
            nseeds = 1, init_mode = nothing, init = ExplicitInit([[-1.0]])), context())
        s = BAT.samplesof(em)
        @test logpdf.(Ref(Distribution(em.approx.transformed)), s.v) ≈ logpdf.(Ref(prior), s.v)
        @test all(iszero, s.weight[first.(s.v) .<= 0])
        @test sum(s.weight) > 0
        recovered = evalmeasure(p, MolewhackerSampling(nsamples = 32, maxiter = 0,
            nseeds = 3, init_mode = nothing, init = ExplicitInit([[-1.0], [-1.0], [1.0]])), context())
        q_recovered = Distribution(recovered.approx.transformed)
        points = [[0.0], [1.0], [3.0]]
        expected = [pdf(Normal(1, sqrt(1 / (1 + exp(1)))), x[1]) for x in points]
        @test pdf.(Ref(q_recovered), points) ≈ expected
        half_target = PosteriorMeasure(Likelihood(z -> Poisson(z[1] > 0 ? 1.0 : 0.0), 1), prior)
        no_mass_round = evalmeasure(half_target, MolewhackerSampling(nsamples = 128, batchsize = 1,
            maxiter = 5, nseeds = 0), BATContext(rng = BAT.Random.Xoshiro(6), ad = ForwardDiff))
        info = no_mass_round.evalinfo.result
        @test info.nevals == 130
        @test info.stop_reason == :no_finite_candidate
    end
end
