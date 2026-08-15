using BAT
using Test

using BATTestCases
using Distributions
using ValueShapes
using IntervalSets
using LinearAlgebra: Diagonal, ones
using Random: Xoshiro


@testset "bridge_sampling_integration" begin
    context = BATContext()

    function test_integration(algorithm::IntegrationAlgorithm, title::String,
                              dist::Distribution; val_expected::Real=1.0,
                              val_rtol::Real=3.5, err_max::Real=0.2)
        @testset "$title" begin
            samplingalg = TransformedMCMC(
                pretransform = DoNotTransform(),
                nsteps = 2*10^5,
                burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 10^5, max_ncycles = 60)
            )
            samples = bat_sample(dist, samplingalg).result

            sd = EvaluatedMeasure(dist, samples = samples)
            sample_integral = bat_integrate(sd, algorithm, context).result

            @test isapprox(sample_integral.val, val_expected, atol=val_rtol*sample_integral.err)
            @test sample_integral.err < err_max
        end
    end

    @testset "weight scale invariance" begin
        function integrate_with_weights(weights)
            context = BATContext(rng=Xoshiro(165))
            dist = MvNormal(zeros(1), ones(1))
            samples = bat_sample(dist, IIDSampling(nsamples=length(weights)), context).result
            samples = DensitySampleVector(samples.v, samples.logd, weight=weights)
            evaluated = EvaluatedMeasure(dist, samples=samples)
            bat_integrate(evaluated, BridgeSampling(), context).result
        end

        uniform_results = integrate_with_weights.((
            ones(3),
            fill(0.75, 3),
            fill(1 / 3, 3),
        ))
        unequal_weights = [1.0, 2.0, 3.0]
        unequal_results = integrate_with_weights.((
            unequal_weights,
            unequal_weights / sum(unequal_weights),
        ))
        long_unequal_weights = collect(1.0:16.0)
        long_unequal_results = integrate_with_weights.((
            long_unequal_weights,
            long_unequal_weights / sum(long_unequal_weights),
        ))

        for result in (uniform_results..., unequal_results..., long_unequal_results...)
            @test isfinite(result.val)
            @test isfinite(result.err)
        end
        for result in uniform_results[2:end]
            @test result.val ≈ uniform_results[1].val rtol=1e-12
            @test result.err ≈ uniform_results[1].err rtol=1e-12
        end
        @test unequal_results[2].val ≈ unequal_results[1].val rtol=1e-12
        @test unequal_results[2].err ≈ unequal_results[1].err rtol=1e-12
        @test long_unequal_results[2].val ≈ long_unequal_results[1].val rtol=1e-12
        @test long_unequal_results[2].err ≈ long_unequal_results[1].err rtol=1e-12
    end

    @testset "finite ESS for short weighted samples" begin
        samples = DensitySampleVector(
            [[0.0], [1.0]],
            zeros(2),
            weight=fill(0.75, 2),
        )
        ess = only(bat_eff_sample_size(samples, EffSampleSizeFromAC(), context).result)

        @test isfinite(ess)
        @test ess > 0
    end

    test_integration(BridgeSampling(pretransform=DoNotTransform()), "funnel distribution", FunnelDistribution(), val_rtol = 15)
    #! ToDo: Fix this test, cause trouble on x86-32
    #test_integration(BridgeSampling(pretransform=DoNotTransform()), "multimodal student-t distribution", MultimodalStudentT(), val_rtol = 50)
    #! ToDo: Fix this test
    # test_integration(BridgeSampling(pretransform=DoNotTransform()), "Gaussian shell", GaussianShell(), val_rtol = 15)
    test_integration(BridgeSampling(pretransform=DoNotTransform()), "MvNormal", MvNormal(Diagonal(ones(5))), val_rtol = 15)
end
