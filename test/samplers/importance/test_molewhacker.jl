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
        ratios = z.logd .- logpdf.(Ref(q), z.v)
        @test z.weight ≈ exp.(ratios .- info.logweight_scale)
        mass_estimate = mean(z.weight) * exp(info.logweight_scale)
        @test mass_estimate ≈ pdf(Normal(0, sqrt(1.25)), 2) rtol = 0.05
    end

    @testset "Reselection preserves fitting multiplicity and budgets" begin
        alg = MolewhackerSampling(nsamples = 64, batchsize = 1, ncandidates = 1,
            nseeds = 0, maxiter = 8, maxcomponents = 4)
        em = evalmeasure(target, alg, context())
        q, info = Distribution(em.approx.transformed), em.evalinfo.result
        @test (info.niterations, info.ncomponents, info.ncomponent_proposals, info.stop_reason) ==
            (3, 2, 4, :maxcomponents)
        @test info.npilot == 1
        # One pool point supplies three selected occurrences of the same Gaussian.
        c = mean(last(q.components))[1]
        a, b = Normal(), Normal(c, sqrt(0.2))
        fitting(x) = (pdf(a, x) + 3pdf(b, x)) / 4
        masses = [pdf(Normal(1.6, sqrt(0.2)), 0) / fitting(0),
            3pdf(Normal(1.6, sqrt(0.2)), c) / fitting(c)]
        masses ./= sum(masses)
        points = [[-1.0], [0.0], [1.6], [3.0]]
        expected = [masses[1] * pdf(a, x[1]) + masses[2] * pdf(b, x[1]) for x in points]
        @test pdf.(Ref(q), points) ≈ expected
    end

    @testset "Fisher covariance and independent factors" begin
        c = atanh(0.5)
        model = z -> MvNormal([2z[1], 2z[1]], [4.0 2tanh(z[1]); 2tanh(z[1]) 1.0])
        p = PosteriorMeasure(Likelihood(model, [0.0, 0.0]), prior)
        seeds = [[c]]
        alg = MolewhackerSampling(nsamples = 32, maxiter = 0, nseeds = 1, init_mode = nothing, init = ExplicitInit(seeds), laplace_seeds = false)
        em = evalmeasure(p, alg, context())
        q = Distribution(em.approx.transformed)
        @test cov(last(q.components))[1, 1] ≈ 4 / 25

        # Compact Normal covariance charts retain the variance-derivative information.
        for (model, variance) in ((z -> MvNormal([2z[1], z[1]], Diagonal(exp.([2z[1], z[1]]))), 2 / 17),
                (z -> MvNormal([2z[1], z[1]], exp(2z[1])I(2)), 1 / 10))
            p_compact = PosteriorMeasure(Likelihood(model, [0.0, 0.0]), prior)
            em_compact = evalmeasure(p_compact, MolewhackerSampling(nsamples = 32, maxiter = 0,
                nseeds = 1, init_mode = nothing, init = ExplicitInit([[0.0]]), laplace_seeds = false), context())
            @test cov(last(Distribution(em_compact.approx.transformed).components))[1, 1] ≈ variance
        end

        # Likelihood information at zero is 1/4 + 2 + 4. Add the prior once.
        product_model = z -> NamedTupleDist(a = Normal(z[1], 2.0), rest = NamedTupleDist(b = Poisson(2exp(z[1])), c = Exponential(3exp(2z[1]))))
        p_product = PosteriorMeasure(Likelihood(product_model, (a = 0.0, rest = (b = 2, c = 3.0))), prior)
        em_product = evalmeasure(p_product, MolewhackerSampling(nsamples = 32, maxiter = 0,
            nseeds = 1, init_mode = nothing, init = ExplicitInit([[0.0]]), laplace_seeds = false), context())
        @test cov(last(Distribution(em_product.approx.transformed).components))[1, 1] ≈ 4 / 29

        # A stationary forward model: Fisher sees only the prior, the target has curvature 1 + 16.
        p_stationary = PosteriorMeasure(Likelihood(z -> Normal(z[1]^2, 0.5), -2.0), prior)
        em_laplace = evalmeasure(p_stationary, MolewhackerSampling(nsamples = 32, maxiter = 0, nseeds = 1,
            init_mode = nothing, init = ExplicitInit([[0.0]]), laplace_seeds = true), context())
        q_laplace = Distribution(em_laplace.approx.transformed)
        @test sort([invcov(c)[1, 1] for c in q_laplace.components]) ≈ [1, 17 / 1.2] rtol = 1e-6
        @test probs(q_laplace) ≈ [0.5, 0.5]
        @test em_laplace.evalinfo.result.nhessians == 1
        # The Newton polish lets the default mode search stop early.
        @test (MolewhackerSampling().init_mode.maxiters, MolewhackerSampling(laplace_seeds = false).init_mode.maxiters) == (50, 1000)

        # Idle tasks split the Jacobian columns. A nonsymmetric map exposes their order.
        A = [1.0 2.0 0.0 -1.0 0.5 0.0; 0.0 1.0 3.0 0.0 -2.0 1.0]
        p_linear = PosteriorMeasure(Likelihood(z -> MvNormal(A * z, Diagonal([0.5, 2.0])), zeros(2)), MvNormal(zeros(6), I(6)))
        em_linear = evalmeasure(p_linear, MolewhackerSampling(nsamples = 32, maxiter = 0, nseeds = 1, init_mode = nothing,
            init = ExplicitInit([zeros(6)]), executor = BAT.MultiThreadedExec(ntasks = 2), laplace_seeds = false), context())
        @test invcov(last(Distribution(em_linear.approx.transformed).components)) ≈ I + A' * Diagonal([2.0, 0.5]) * A

        a, b = [-0.5], [1.5]
        repeated = [a, a, b]
        saved = deepcopy(repeated)
        repeated_result = evalmeasure(target, MolewhackerSampling(nsamples = 32, maxiter = 0,
            nseeds = 3, init_mode = nothing, init = ExplicitInit(repeated), laplace_seeds = false), context())
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
            nseeds = 1, init_mode = nothing, init = ExplicitInit([(rate = 1.0,)]), laplace_seeds = false), context(73))
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
        # Gaussian proposal produced by mode initialization. Fresh rounds draw from
        # the whole mixture, whose Float32 masses shift at rounding level with scale.
        scale_alg = MolewhackerSampling(nsamples = 256, batchsize = 128, maxiter = 2, nseeds = 0, fresh_rounds = 0)
        a = evalmeasure(target, scale_alg, context(74, precision = Float32))
        shifted = evalmeasure(weightedmeasure(1e8, BAT.batmeasure(target)), scale_alg, context(74, precision = Float32))
        @test BAT.samplesof(a).v ≈ BAT.samplesof(shifted).v rtol = 1e-6
        @test BAT.samplesof(a).weight ≈ BAT.samplesof(shifted).weight rtol = 1e-6
        @test BAT.samplesof(a).logd ≈ logdensityof.(Ref(BAT.unevaluated(a)), BAT.samplesof(a).v)
    end

    @testset "Mixture refit from fresh draws" begin
        # From the prior, one fresh round and the weighted EM fit recover this Gaussian
        # posterior: efficiency 0.95-0.98 over six seeds, against 0.63-0.86 without them.
        refit(k; kw...) = evalmeasure(target, MolewhackerSampling(; nsamples = 2000, batchsize = 500, maxiter = 1,
            nseeds = 0, fresh_rounds = k, kw...), context(72)).evalinfo.result
        with, without = refit(1), refit(0)
        @test with.efficiency > 0.93 > without.efficiency
        # Fourteen fresh candidates and one to six fitted Gaussians.
        @test with.ncomponents - without.ncomponents - 14 in 1:6
        @test refit(1; refit = nothing).ncomponents - without.ncomponents == 14
        @test refit(1; refit = MolewhackerRefit(maxcomponents = 1)).ncomponents - without.ncomponents == 15
    end

    @testset "Cached pool scoring" begin
        # A second round adds points and components. The cache matches full scoring,
        # and a zero limit takes the uncached path.
        rng = StableRNG(3)
        comps = [BAT._mw_gaussian(randn(rng, 4), let A = randn(rng, 4, 4); A'A / 4 + I end) for _ in 1:12]
        x1, x = randn(rng, 4, 50), randn(rng, 4, 80)
        x[:, 1:50] = x1
        w = rand(rng, 12)
        w[3] = 0
        q1, q2 = MixtureModel(comps[1:7], w[1:7] ./ sum(w[1:7])), MixtureModel(comps, w ./ sum(w))
        ex = BAT.MultiThreadedExec(ntasks = 2)
        l1, cache = BAT._mw_pool_logpdf(q1, comps[1:7], x1, zeros(0, 0), ex)
        l2, cache = BAT._mw_pool_logpdf(q2, comps, x, cache, ex)
        @test l1 ≈ BAT._mw_batched_logpdf(q1, x1) && l2 ≈ BAT._mw_batched_logpdf(q2, x) && size(cache) == (80, 12)
        @test first(BAT._mw_pool_logpdf(q2, comps, x, cache, ex, 0)) ≈ l2
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
        # The prior proposal matches this flat target, so all weights are equal.
        @test sized.evalinfo.result.max_weight ≈ 1 / length(BAT.samplesof(sized))
        budgets = evalmeasure(flat, MolewhackerSampling(nsamples = 256, target_ess = 32,
            batchsize = 64, maxiter = 1, nseeds = 0), context())
        @test budgets.evalinfo.result.niterations == 1
        for (rule, reason) in (((; target_pool_ess = 32), :pilot_ess),
                ((; target_efficiency = 128 / 256), :pool_efficiency))
            stopped = evalmeasure(flat, MolewhackerSampling(; nsamples = 256, target_ess = 128,
                batchsize = 64, maxiter = 10, nseeds = 0, fresh_rounds = 0, rule...), context())
            info = stopped.evalinfo.result
            @test (info.niterations, info.ncomponents, info.stop_reason) == (0, 1, reason)
            @test info.ess ≈ 128
            # Fresh rounds still follow a threshold stop.
            refreshed = evalmeasure(flat, MolewhackerSampling(; nsamples = 256, batchsize = 64, maxiter = 10,
                nseeds = 0, fresh_rounds = 1, rule...), context()).evalinfo.result
            @test (refreshed.niterations, refreshed.nfresh, refreshed.stop_reason) == (0, 1, reason)
        end

        concentrated = PosteriorMeasure(Likelihood(z -> MvNormal(z, 0.25I(18)), fill(2.0, 18)),
            MvNormal(zeros(18), I(18)))
        seeded = evalmeasure(concentrated, MolewhackerSampling(nsamples = 1000, maxiter = 0, maxcomponents = 20), context())
        @test seeded.evalinfo.result.efficiency > 0.8
        @test maximum(abs, mean(BAT.samplesof(seeded)) .- 1.6) < 0.06
        limited = evalmeasure(target, MolewhackerSampling(nsamples = 64, maxiter = 0, nseeds = 1,
            init = ExplicitInit([[0.0]]), init_mode = OptimAlg(optalg = Optim.LBFGS()), maxevals = 66), context())
        @test limited.evalinfo.result.nevals <= 66
        @test limited.evalinfo.result.nseed_exhausted == 1
        # A fresh round follows adaptation and adds a proposal batch before selection.
        fresh = evalmeasure(target, MolewhackerSampling(nsamples = 32, batchsize = 64, maxiter = 1,
            fresh_rounds = 1, nseeds = 0), context()).evalinfo.result
        h = fresh.history
        @test (fresh.niterations, fresh.nfresh, getproperty.(h, :fresh)) == (1, 1, [false, true])
        @test (h[1].npilot - h[1].drawn, h[2].npilot - h[2].drawn) == (64, h[1].npilot + 64)
    end

    @testset "Defensive nonlinear tails" begin
        p = PosteriorMeasure(Likelihood(z -> Normal(2tanh(z[1]), 1.0), 0.0), prior)
        ε = 0.1
        em = evalmeasure(p, MolewhackerSampling(nsamples = 16, maxiter = 0,
            nseeds = 1, init_mode = nothing, init = ExplicitInit([[0.0]]), laplace_seeds = false, exploration_mass = ε), context())
        q = Distribution(em.approx.transformed)
        points = [[0.0], [-12.0], [12.0]]
        # Fisher variance at zero is 1/5. This local Gaussian alone has infinite IS variance.
        logw = logdensityof.(Ref(BAT.batmeasure(p)), points) .- logpdf.(Ref(q), points)
        @test all(logw .<= logpdf(Normal(), 0.0) - log(ε))

        # The local Gaussian alone gives unbounded weights exp(2z²), a heavy Pareto tail.
        # Prior mixing bounds them, so the fitted shape turns negative. PSIS changes
        # only the largest raw ratios.
        n = 2000
        tails(; kw...) = evalmeasure(p, MolewhackerSampling(; nsamples = n, maxiter = 0, nseeds = 1,
            init_mode = nothing, init = ExplicitInit([[0.0]]), laplace_seeds = false, kw...), context(75))
        raw, mixed, smoothed = tails(), tails(exploration_mass = ε), tails(smooth_weights = true)
        @test mixed.evalinfo.result.pareto_k < 0 < raw.evalinfo.result.pareto_k
        logratio(em) = log.(BAT.samplesof(em).weight) .+ em.evalinfo.result.logweight_scale
        r, s = logratio(raw), logratio(smoothed)
        tail = partialsortperm(r, 1:ceil(Int, 3sqrt(n)), rev = true)
        @test r[setdiff(eachindex(r), tail)] ≈ s[setdiff(eachindex(s), tail)]
        @test maximum(s) <= maximum(r) && issorted(s[reverse(tail)])
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
            nseeds = 3, init_mode = nothing, init = ExplicitInit([[-1.0], [-1.0], [1.0]]), laplace_seeds = false), context())
        q_recovered = Distribution(recovered.approx.transformed)
        points = [[0.0], [1.0], [3.0]]
        expected = [pdf(Normal(1, sqrt(1 / (1 + exp(1)))), x[1]) for x in points]
        @test pdf.(Ref(q_recovered), points) ≈ expected
        half_target = PosteriorMeasure(Likelihood(z -> Poisson(z[1] > 0 ? 1.0 : 0.0), 1), prior)
        no_mass_round = evalmeasure(half_target, MolewhackerSampling(nsamples = 128, batchsize = 1,
            maxiter = 5, nseeds = 0), BATContext(rng = BAT.Random.Xoshiro(6), ad = ForwardDiff))
        info = no_mass_round.evalinfo.result
        # One fresh draw follows the stop and finds no mass either.
        @test (info.nevals, info.nfresh) == (131, 1)
        @test info.stop_reason == :no_finite_candidate
    end
end
