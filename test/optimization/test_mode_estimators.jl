using BAT
using Test

using LinearAlgebra, Distributions, StatsBase, ValueShapes, Random123, DensityInterface
using UnPack, InverseFunctions
using AutoDiffOperators, ForwardDiff
using Optim

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
            #    f_logd = logdensityof(posterior) ∘ inverse(res.trafo)
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


    @testset "MaxDensitySearch" begin
        context = BATContext()
        @test @inferred(bat_findmode(samples, MaxDensitySearch(), context)).result isa NamedTuple
        m = bat_findmode(samples, MaxDensitySearch(), context)
        @test samples[m.mode_idx].v == m.result
        @test isapprox(unshaped(m.result, elshape(samples.v)), true_mode_flat, rtol = 0.05)
    end


    @testset "NelderMead" begin
        context = BATContext(rng = Philox4x((0, 0)))
        test_findmode(posterior, OptimAlg(optalg = NelderMead(), trafo = DoNotTransform()), 0.01, context)
    end


    @testset "LBFGS" begin
        context = BATContext(rng = Philox4x((0, 0)), ad = ADModule(:ForwardDiff))
        # Result Optim.maximize with LBFGS is not type-stable:
        test_findmode(posterior, OptimAlg(optalg = LBFGS(), trafo = DoNotTransform()), 0.01, inferred = false, context)

        test_findmode_ctx(posterior, OptimAlg(optalg = LBFGS(), trafo = DoNotTransform()), 0.01, context)
    end
end
