using BAT
using Test

using AutoDiffOperators
using LinearAlgebra, Distributions, StatsBase, ValueShapes, Random, Random123, DensityInterface
using UnPack, InverseFunctions
import ForwardDiff
using Optim, OptimizationOptimJL, OptimizationLBFGSB

@testset "mode_estimators" begin
    prior = NamedTupleDist(
        x = Normal(2.0, 1.0),
        c = [4, 5],
        a = MvNormal([1.5, 0.5, 2.5], Matrix{Float32}(I, 3, 3))
    )

    posterior = PosteriorMeasure(logfuncdensity(v -> 0), prior)

    true_mode_flat = [2.0, 1.5, 0.5, 2.5]
    true_mode = varshape(prior)(true_mode_flat)

    samples = @inferred(bat_sample(prior, IIDSampling(nsamples = 10^5), BATContext())).result


    function test_findmode(posterior, algorithm, rtol, context::BATContext; inferred::Bool = true)
        @testset "test_findmode $(nameof(typeof(algorithm)))" begin
            res = if inferred
                @inferred(bat_findmode(posterior, algorithm, context))
            else
                (bat_findmode(posterior, algorithm, context))
            end
            @test keys(res.result) == keys(true_mode)
            @test isapprox(unshaped(res.result, varshape(posterior)), true_mode_flat, rtol = rtol)
            
            # # ToDo: Re-enable trace tests once tracing has been re-enabled:
            #if hasproperty(res.trace_trafo, :grad_logd)
            #    @unpack v, logd, grad_logd = res.trace_trafo
            #    f_logd = logdensityof(posterior) ∘ inverse(res.f_pretransform)
            #    @test all(f_logd.(v) .≈ logd)
            #    @test all(grad_logd .≈ ForwardDiff.gradient.(Ref(f_logd), v))
            #else
            #    @test hasproperty(res.trace_trafo, :v)
            #end
        end
    end

    function test_findmode_ctx(posterior, algorithm, rtol, context)
        res = (bat_findmode(posterior, algorithm, context))
        @test keys(res.result) == keys(true_mode)
        @test isapprox(unshaped(res.result, varshape(posterior)), true_mode_flat, rtol = rtol)
    end


    @testset "ModeAsDefined" begin
        context = BATContext()
        @test @inferred(bat_findmode(prior, ModeAsDefined(), context)).result == true_mode
        @test @inferred(bat_findmode(batmeasure(prior), ModeAsDefined(), context)).result == true_mode
        let post_modes = @inferred(bat_findmode(posterior, context)).result
            for k in keys(post_modes)
                @test isapprox(post_modes[k], true_mode[k], atol=0.001)
            end
        end
    end


    @testset "bat_bgml" begin
        context = BATContext(ad = ForwardDiff)
        algorithm = TransformedMaxDensity(optalg = OptimAlg(optalg = LBFGS()), init = ExplicitInit([(mu = 0.2,)]))
        likelihood = logfuncdensity(p -> logpdf(Normal(p.mu, 0.5), 1.2))
        flat_prior = distprod(mu = Uniform(-2, 4))
        informative_prior = distprod(mu = Normal(-1.0, 0.03))

        # Only the likelihood is maximized, so the prior must not shift the result:
        for pr in (flat_prior, informative_prior)
            r = bat_bgml(likelihood, pr, algorithm, context)
            @test r.result.mu ≈ 1.2 atol = 1e-3
            # The result is a likelihood maximizer, it must not be recorded
            # as a mode of an evaluated posterior measure:
            @test !haskey(r, :evaluated)
        end

        # The mode of the posterior does follow an informative prior:
        r_map = bat_findmode(PosteriorMeasure(likelihood, informative_prior), algorithm, context)
        @test r_map.result.mu ≈ -0.99 atol = 0.01

        # With a pretransform that maps the search space onto the support of
        # the prior, the result stays within that support even if the
        # likelihood peaks outside of it:
        bounded_prior = distprod(mu = Uniform(-1, 1))
        outside_lik = logfuncdensity(p -> logpdf(Normal(3, 1), p.mu))
        for alg in (
            TransformedMaxDensity(optalg = OptimAlg(optalg = LBFGS()), init = ExplicitInit([(mu = 0.2,)])),
            TransformedMaxDensity(optalg = OptimizationAlg(optalg = OptimizationOptimJL.NelderMead()), init = ExplicitInit([(mu = 0.2,)])),
        )
            r_b = bat_bgml(outside_lik, bounded_prior, alg, context)
            @test -1 <= r_b.result.mu <= 1
            @test r_b.result.mu ≈ 1 atol = 0.05
        end

        # Without a space transformation only the likelihood shapes the
        # result, the prior does not constrain it:
        alg_notrafo = TransformedMaxDensity(optalg = OptimAlg(optalg = Optim.NelderMead()), pretransform = DoNotTransform(), init = ExplicitInit([(mu = 0.2,)]))
        r_u = bat_bgml(outside_lik, bounded_prior, alg_notrafo, context)
        @test r_u.result.mu ≈ 3 atol = 0.05

        # bat_bgml shares the default mode estimator of bat_findmode:
        @test BAT.bat_default(bat_bgml, Val(:algorithm), likelihood, flat_prior) isa TransformedMaxDensity
        r_default = bat_bgml(likelihood, flat_prior, context)
        @test r_default.result.mu ≈ 1.2 atol = 0.01
    end


    @testset "MaxDensitySearch" begin
        context = BATContext()
        @test @inferred(bat_findmode(samples, MaxDensitySearch(), context)).result isa NamedTuple
        m = bat_findmode(samples, MaxDensitySearch(), context)
        @test samples[m.mode_idx].v == m.result
        @test isapprox(unshaped(m.result, elshape(samples.v)), true_mode_flat, rtol = 0.05)
    end


    @testset "Optim.jl - NelderMead" begin
        context = BATContext(rng = Philox4x((0, 0)))
        test_findmode(posterior, TransformedMaxDensity(optalg = OptimAlg(optalg = NelderMead()), pretransform = DoNotTransform()), 0.01, context)
    end

    @testset "Optim.jl with custom options" begin # checks that options are correctly passed to Optim.jl
        context = BATContext(rng = Philox4x((0, 0)))
        optimizer = TransformedMaxDensity(optalg = OptimAlg(optalg = NelderMead(), maxiters=20, maxtime=30, reltol=0.2, kwargs=(f_calls_limit=25,)), pretransform = DoNotTransform())
        
        result = bat_findmode(posterior, optimizer, context)
        @test result.info.iterations <= 20
        @test result.info.time_limit == 30
        @test result.info.f_reltol == 0.2
        @test result.info.f_calls <= 26

    end

    @testset "algorithm auto-wrapping" begin
        context = BATContext(rng = Philox4x((0, 0)))
        @test @inferred(BAT.batalgorithm(OptimAlg(optalg = NelderMead()))) isa TransformedMaxDensity
        @test @inferred(BAT.batalgorithm(Optim.NelderMead())) isa TransformedMaxDensity
        # Bare backends and raw Optim optimizers work as mode estimators:
        test_findmode_ctx(posterior, OptimAlg(optalg = NelderMead()), 0.01, context)
        test_findmode_ctx(posterior, Optim.NelderMead(), 0.01, context)
    end

    @testset "Optim.jl - LBFGS" begin
        context = BATContext(rng = Philox4x((0, 0)), ad = ForwardDiff)
        # Result Optim.maximize with LBFGS is not type-stable:
        test_findmode(posterior, TransformedMaxDensity(optalg = OptimAlg(optalg = LBFGS()), pretransform = DoNotTransform()), 0.01, inferred = false, context)

        test_findmode_ctx(posterior, TransformedMaxDensity(optalg = OptimAlg(optalg = LBFGS()), pretransform = DoNotTransform()), 0.01, context)
    end


    @testset "OptimizationBase.jl" begin
        context = BATContext(rng = Philox4x((0, 0)))
        # result is not type-stable:
        test_findmode(posterior, TransformedMaxDensity(optalg = OptimizationAlg(optalg = OptimizationOptimJL.NelderMead()), pretransform = DoNotTransform()), 0.01, context, inferred = false) 

        context = BATContext(rng = Philox4x((0, 0)), ad = ADSelector(ForwardDiff))
        # result is not type-stable:
        test_findmode(posterior, TransformedMaxDensity(optalg = OptimizationAlg(optalg = OptimizationLBFGSB.LBFGSB()), pretransform = DoNotTransform()), 0.01, context, inferred = false) 
    end

    @testset "OptimizationBase.jl with custom options" begin # checks that options are correctly passed to OptimizationBase.jl
        # ParticleSwarm draws from the global RNG, not from the context RNG,
        # so seed it for deterministic results:
        Random.seed!(0x424154)
        context = BATContext(rng = Philox4x((0, 0)))
        optimizer = TransformedMaxDensity(optalg = OptimizationAlg(optalg = OptimizationOptimJL.ParticleSwarm(n_particles=10), maxiters=200, kwargs=(f_calls_limit=500,)), pretransform = DoNotTransform())

        # result is not type-stable:
        test_findmode(posterior, optimizer, 0.01, context, inferred = false) 

        optimizer = TransformedMaxDensity(optalg = OptimizationAlg(optalg = OptimizationOptimJL.ParticleSwarm(n_particles=10),
            maxiters=200, maxtime=30, reltol=0.2, kwargs=(f_calls_limit=500,)), pretransform = DoNotTransform())

        result = bat_findmode(posterior, optimizer, context)
        @test result.info.cache.solver_args.maxiters == 200
        @test result.info.cache.solver_args.f_calls_limit == 500
        @test result.info.cache.solver_args.reltol == 0.2
        @test result.info.cache.solver_args.maxtime == 30
        @test result.info.original.method.n_particles == 10
    end

end
