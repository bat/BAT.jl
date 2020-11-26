using BAT
using Test

using Distributions
using ValueShapes
using IntervalSets
using LinearAlgebra: Diagonal, ones


@testset "ahmi_integration" begin
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
    test_integration(AHMIntegration(), "funnel distribution", BAT.FunnelDistribution(), val_rtol = 14)
    test_integration(AHMIntegration(), "multimodal cauchy", BAT.MultimodalCauchy(), val_rtol = 35)
    test_integration(AHMIntegration(), "Gaussian shell", BAT.GaussianShell(), val_rtol = 6.2)
    test_integration(AHMIntegration(), "MvNormal", MvNormal(Diagonal(ones(5))), val_rtol = 15)

    @testset "ahmi_integration_defaults" begin
        bsample = @inferred(bat_sample(product_distribution([Normal() for i in 1:3]), 10^4)).result
        @test isapprox(bat_integrate(bsample).result.val, 1.0, rtol=15)
        eff_sample_size_dsv = @inferred(bat_eff_sample_size(bsample)).result
        eff_sample_size_aosa = @inferred(bat_eff_sample_size(bsample.v)).result

        [bsample.weight[i] = 0 for i in eachindex(bsample.weight)]
        bsample.weight[1] = 1
        eff_sample_size_dsv_single_weight = @inferred(bat_eff_sample_size(bsample)).result

        for i in eachindex(@inferred(bat_eff_sample_size(bsample)).result)
            @test 8000 <= eff_sample_size_dsv[i] <= 10^4
            @test eff_sample_size_dsv[i] == eff_sample_size_aosa[i]
        end

        for i in eachindex(eff_sample_size_dsv_single_weight)
            @test @inferred(isapprox(1, eff_sample_size_dsv_single_weight[i]))
        end

    end

end
