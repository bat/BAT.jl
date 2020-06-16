using Distributions
using BAT
using ValueShapes
using IntervalSets
using Test

@testset "bat_integrate" begin
    @testset "multimodal cauchy" begin
        μ=1.0
        σ=0.2
        n_modes=10

        lb = -8
        ub = 8

        dist = BAT.MultimodalCauchy(μ, σ, n_modes) 
        dist = Distributions.truncated(dist, lb, ub)

        @test size(dist) == (n_modes,)

        sample = bat_sample(dist, 100000).result
        sample_integral = bat_integrate(sample).result

        @test isapprox(sample_integral.val, 1, atol=3*sample_integral.err)
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

        @test isapprox(sample_integral, 1., atol=3*sample_integral.err)
    end

    @testset "funnel" begin
        dist = BAT.Funnel(1.0, 0.5, [1.0, 2.0, 3.0])
        trunc_dist = truncated(dist, -50, 50)

        @test mean(trunc_dist) == [0.0, 0.0, 0.0]

        @test size(dist) == (3,)

        sample = bat_sample(trunc_dist, 100000).result
        sample_integral = bat_integrate(sample).result

        @test isapprox(sample_integral.val, 1, atol=3*sample_integral.err)
    end
end
