# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Distributions, StatsBase


@testset "importance_samplers" begin
    function test_moments(dist::AnyDensityLike, algo::BAT.AbstractSamplingAlgorithm; rtol::Real=0.01)
        samples = @inferred(bat_sample(dist, algo)).result

        @test isapprox(@inferred(mean(samples)), @inferred(mean(dist)), rtol=rtol)
        @test isapprox(@inferred(var(samples)), @inferred(var(dist)), rtol=rtol)
    end

    @testset "sobol_sampler" begin
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])
        algo = SobolSampler(nsamples=10^5)

        test_moments(dist, algo)
    end

    @testset "grid_sampler" begin
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])
        algo = GridSampler(ppa=500)

        # ToDO: Use @inferred when type stable
        samples = bat_sample(dist, algo).result
        
        @test isapprox(@inferred(mean(samples)), @inferred(mean(dist)), rtol=0.1)
        @test isapprox(@inferred(var(samples)), @inferred(var(dist)), rtol=0.1)
    end
end
