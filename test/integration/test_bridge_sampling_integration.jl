using BAT
using Test

using BATTestCases
using Distributions
using ValueShapes
using IntervalSets
using LinearAlgebra: Diagonal, ones
import Measurements


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
