# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Distributions, StatsBase


@testset "bat_sample" begin
    @testset "IIDSampling" begin
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])

        @test length(@inferred(bat_sample(dist, IIDSampling(nsamples = 10^3))).result) == 10^3

        @test @inferred(bat_sample(Random.GLOBAL_RNG, dist)).result isa DensitySampleVector
        @test @inferred(bat_sample(dist, BAT.IIDSampling())).result isa DensitySampleVector
        @test @inferred(bat_sample(Random.GLOBAL_RNG, dist, BAT.IIDSampling())).result isa DensitySampleVector

        samples = @inferred(bat_sample(dist, IIDSampling(nsamples = 10^5))).result
        @test isapprox(mean(samples.v), [0.4, 0.6]; rtol = 0.05)
        @test isapprox(cov(samples.v), [2.0 1.2; 1.2 3.0]; rtol = 0.05)
        @test all(isequal(1), samples.weight)
        
        resamples = @inferred(bat_sample(samples, OrderedResampling(nsamples = length(samples)))).result
        @test samples == resamples

        dist_bmode = @inferred(bat_findmode(dist)).result
        @test @inferred(length(dist_bmode)) == 2

        dist_sample_vector_bmode = @inferred(bat_findmode(samples)).result
        @test @inferred(length(dist_sample_vector_bmode)) == 2

        isapprox(var(bat_sample(Normal(), BAT.IIDSampling(nsamples = 10^3)).result), [1], rtol = 10^-1)
    end
end
