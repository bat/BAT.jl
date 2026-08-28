using BAT
using Test

using BATTestCases
using Distributions
using ValueShapes
using IntervalSets
using Random123
using LinearAlgebra: Diagonal, ones
import Measurements


@testset "bridge_sampling_integration" begin
    context = BATContext(rng = Philox4x((564, 82)))

    @testset "automatic proposal count" begin
        dist = MvNormal([0.0], ones(1, 1))
        target = batmeasure(dist)
        v = [[-1.5], [-0.5], [0.5], [1.5], [-1.0], [-0.25], [0.75], [1.25]]
        logd = map(x -> logpdf(dist, x), v)

        function run_bridge(weights, seed)
            samples = DensitySampleVector(v = v, logd = logd, weight = weights)
            context = BATContext(rng = Philox4x(seed))
            result = BAT.bridge_sampling_integral(
                target, samples, false, EffSampleSizeFromAC(), context
            )
            result, rand(context.rng)
        end

        function expected_rng(weights, n, seed)
            samples = DensitySampleVector(v = v, logd = logd, weight = weights)
            first_batch = samples[1:4]
            proposal = batmeasure(MvNormal(vec(mean(first_batch)), Array(cov(first_batch))))
            context = BATContext(rng = Philox4x(seed))
            evalmeasure(proposal, IIDSampling(nsamples = n), context)
            rand(context.rng)
        end

        count_cases = (
            ("ordinary", [ones(4); [1.0, 1.0, 4.0, 4.0]], 3),
            ("integer extreme", [ones(Int, 4); [typemax(Int), 1, 1, 1]], 1),
            ("floating extreme", [ones(4); fill(floatmax(Float64), 4)], 4),
            ("subnormal scale", [ones(4); fill(nextfloat(0.0), 4)], 4),
            ("huge BigFloat", [ones(BigFloat, 4); fill(big"1e1000", 4)], 4),
        )
        for (i, (name, weights, expected_count)) in enumerate(count_cases)
            @testset "$name weights" begin
                seed = (564, 820 + i)
                _, next_rng = run_bridge(weights, seed)
                @test next_rng == expected_rng(weights, expected_count, seed)
            end
        end

        weights = [ones(4); [1.0, 1.0, 4.0, 4.0]]
        result, next_rng = run_bridge(weights, (564, 830))
        scaled_result, scaled_next_rng = run_bridge(1e100 .* weights, (564, 830))
        @test scaled_result[1] ≈ result[1]
        @test scaled_result[2] ≈ result[2]
        @test scaled_next_rng == next_rng
    end

    function test_integration(algorithm::IntegrationAlgorithm, title::String,
                              dist::Distribution; val_expected::Real=1.0,
                              val_rtol::Real=3.5, err_max::Real=0.2)
        @testset "$title" begin
            samplingalg = TransformedMCMC(
                pretransform = DoNotTransform(),
                nsteps = 2*10^5,
                burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 10^5, max_ncycles = 60)
            )
            samples = bat_sample(dist, samplingalg, context).result

            sd = EvaluatedMeasure(dist, empirical = samples)
            # Masses/integrals are reported on the canonical logarithmic scale:
            logint = log(bat_integrate(sd, algorithm, context).result)

            @test isapprox(Measurements.value(logint), log(val_expected), atol = val_rtol * Measurements.uncertainty(logint))
            @test Measurements.uncertainty(logint) < err_max
        end
    end

    @testset "non-integer weights" begin
        dist = MvNormal(zeros(2), ones(2))
        samples = bat_sample(dist, IIDSampling(nsamples=20), context).result
        samples = DensitySampleVector(v = samples.v, logd = samples.logd, weight = fill(0.75, length(samples)))
        evaluated = EvaluatedMeasure(dist, empirical = samples)

        result = bat_integrate(evaluated, BridgeSampling(pretransform=DoNotTransform()), context).result
        @test isfinite(Measurements.value(log(result)))
    end

    test_integration(BridgeSampling(pretransform=DoNotTransform()), "funnel distribution", FunnelDistribution(), val_rtol = 15)
    #! ToDo: Fix this test, cause trouble on x86-32
    #test_integration(BridgeSampling(pretransform=DoNotTransform()), "multimodal student-t distribution", MultimodalStudentT(), val_rtol = 50)
    #! ToDo: Fix this test
    # test_integration(BridgeSampling(pretransform=DoNotTransform()), "Gaussian shell", GaussianShell(), val_rtol = 15)
    test_integration(BridgeSampling(pretransform=DoNotTransform()), "MvNormal", MvNormal(Diagonal(ones(5))), val_rtol = 15)
end
