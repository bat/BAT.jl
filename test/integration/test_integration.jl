using Distributions
using BAT
using ValueShapes
using IntervalSets
using LinearAlgebra: Diagonal, ones
using Test

@testset "bat_integrate" begin
    function test_integration(algorithm::IntegrationAlgorithm, title::String,
                              dist::Distribution; nsamples::Integer=10^5, val_expected::Real=1.0,
                              val_rtol::Real=3.5, err_max::Real=0.2)
        @testset "$title" begin
            sample = bat_sample(dist, nsamples).result
            sample_integral = bat_integrate(sample, algorithm).result

            @test isapprox(sample_integral.val, val_expected, atol=val_rtol*sample_integral.err)
            @test sample_integral.err < err_max
        end
    end
    test_integration(AHMIntegration(), "funnel distribution", BAT.FunnelDistribution(), val_rtol=5.0)
    test_integration(AHMIntegration(), "multimodal cauchy", BAT.MultimodalCauchy(), val_rtol = 35.0)
    test_integration(AHMIntegration(), "MvNormal", MvNormal(Diagonal(ones(5))), val_rtol=15)
end
