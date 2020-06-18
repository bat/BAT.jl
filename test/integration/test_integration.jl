using Distributions
using BAT
using ValueShapes
using IntervalSets
using Test

@testset "bat_integrate" begin
    @testset "multimodal cauchy" begin
        dist = BAT.MultimodalCauchy() 

        @test size(dist) == (4,)

        sample = bat_sample(dist, 100000).result
        sample_integral = bat_integrate(sample).result

        @test isapprox(sample_integral.val, 1, atol=3.1*sample_integral.err)
        @test sample_integral.err < 0.15
    end

    @testset "MvN" begin
        dist = MvNormal([1. 0. 0. 0. 0.;
                         0. 1. 0. 0. 0.;
                         0. 0. 1. 0. 0.;
                         0. 0. 0. 1. 0.;
                         0. 0. 0. 0. 1.])
        sample = bat_sample(dist, 100000).result
        sample_integral = bat_integrate(sample).result

        @test isapprox(sample_integral, 1., atol=3.1*sample_integral.err)
    end

    @testset "funnel" begin
        dist = BAT.FunnelDistribution()

        @test mean(dist) == [0.0, 0.0, 0.0]

        @test size(dist) == (3,)

        sample = bat_sample(dist, 100000).result
        sample_integral = bat_integrate(sample).result

        @test isapprox(sample_integral.val, 1, atol=3.1*sample_integral.err)
    end
end
