using Distributions
using BAT
using ValueShapes
using IntervalSets
using LinearAlgebra: Diagonal, ones
using Test

@testset "bat_integrate" begin
    function test_integration(algorithm::IntegrationAlgorithm, title::String, 
                              dist::Distribution, nsamples::Int=10^5, val_expected::Real=1.0, 
                              val_rtol::Real=3.1, err_max::Real=0.1)
        @testset "$title" begin
            sample = bat_sample(dist, nsamples).result
            sample_integral = bat_integrate(sample, algorithm).result

            @test isapprox(sample_integral.val, val_expected, atol=val_rtol*sample_integral.err)
            @test sample_integral.err < err_max
        end
    end
    test_integration(AHMIntegration(), "multimodal cauchy", BAT.MultimodalCauchy(), 10^5)
    test_integration(AHMIntegration(), "funnel distribution", BAT.FunnelDistribution(), 10^5)
    test_integration(AHMIntegration(), "MvNormal", MvNormal(Diagonal(ones(5))), 10^5)
end
