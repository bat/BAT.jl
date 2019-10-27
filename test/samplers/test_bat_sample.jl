# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Distributions, StatsBase


@testset "bat_sample" begin
    @testset "RandSampling" begin
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])

        @test length(@inferred(bat_sample(dist, 1)).samples) == 1
        @test length(@inferred(bat_sample(dist, 10^3)).samples) == 10^3

        @test length(@inferred(bat_sample(Random.GLOBAL_RNG, dist, 10^2)).samples) == 10^2
        @test length(@inferred(bat_sample(dist, 10^2, BAT.RandSampling())).samples) == 10^2
        @test length(@inferred(bat_sample(Random.GLOBAL_RNG, dist, 10^2, BAT.RandSampling())).samples) == 10^2

        samples, stats = bat_sample(dist, 10^5)
        @test isapprox(mean(samples.params), [0.4, 0.6]; rtol = 0.05)
        @test isapprox(cov(samples.params), [2.0 1.2; 1.2 3.0]; rtol = 0.05)
        @test all(isequal(0), samples.log_prior)
        @test all(isequal(1), samples.weight)
        @test stats isa NamedTuple{(:mode,:mean,:cov)}
        @test stats == bat_stats(samples)
    end
end
