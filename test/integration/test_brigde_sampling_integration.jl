using BAT
using Test

using Distributions
using ValueShapes
using IntervalSets
using LinearAlgebra: Diagonal, ones


@testset "bridge_sampling_integration" begin
    function test_integration(algorithm::IntegrationAlgorithm, title::String,
                              dist::Distribution; val_expected::Real=1.0,
                              val_rtol::Real=3.5, err_max::Real=0.2)
        @testset "$title" begin
            samplingalg = MCMCSampling(
                mcalg = MetropolisHastings(),
                trafo = NoDensityTransform(),
                nsteps = 2*10^5,
                burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 10^5, max_ncycles = 60)
            )
            sample = bat_sample(dist, samplingalg).result
            sd = SampledDensity(dist, sample)
            sample_integral = bat_integrate(sd, algorithm).result

            @test isapprox(sample_integral.val, val_expected, atol=val_rtol*sample_integral.err)
            @test sample_integral.err < err_max
        end
    end
    test_integration(BridgeSampling(trafo=NoDensityTransform()), "funnel distribution", BAT.FunnelDistribution(), val_rtol = 15)
    test_integration(BridgeSampling(trafo=NoDensityTransform()), "multimodal student-t distribution", BAT.MultimodalStudentT(), val_rtol = 50)
    test_integration(BridgeSampling(trafo=NoDensityTransform()), "Gaussian shell", BAT.GaussianShell(), val_rtol = 15)
    test_integration(BridgeSampling(trafo=NoDensityTransform()), "MvNormal", MvNormal(Diagonal(ones(5))), val_rtol = 15)

end