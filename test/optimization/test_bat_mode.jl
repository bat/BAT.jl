# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Distributions, StatsBase, ValueShapes

@testset "bat_mode" begin
    prior = NamedTupleDist(
        x = Normal(2.0, 1.0),
        c = [4, 5],
        a = MvNormal([1.5, 0.5, 2.5], Matrix{Float32}(I(3)))
    )

    posterior = PosteriorDensity(X -> 0, prior)

    true_mode = [2.0, 1.5, 0.5, 2.5]

    samples = @inferred bat_sample(prior, 10^5)


    @testset "ModeAsDefined" begin
        @test @inferred(BAT.default_mode_estimator(prior)) == ModeAsDefined()
        @test @inferred(bat_mode(prior)).mode == true_mode
        @test @inferred(bat_mode(BAT.DistributionDensity(prior), ModeAsDefined())).mode == true_mode
    end


    @testset "MaxDensitySampleSearch" begin
        @test @inferred(BAT.default_mode_estimator(samples.samples)) == MaxDensitySampleSearch()
        @test @inferred(bat_mode(samples.samples, MaxDensitySampleSearch())) == @inferred(bat_mode(samples.samples))
        m = bat_mode(samples.samples, MaxDensitySampleSearch())
        @test samples.samples[m.mode_idx].params == m.mode
        @test isapprox(m.mode, true_mode, rtol = 0.05)
    end


    @testset "MaxDensityNelderMead" begin
        @test @inferred(BAT.default_mode_estimator(posterior)) == MaxDensityNelderMead()
        @test isapprox(@inferred(bat_mode(posterior, MaxDensityNelderMead())).mode, true_mode, rtol = 0.01)
        @test isapprox(@inferred(bat_mode(posterior, MaxDensityNelderMead(), initial_mode = rand(prior))).mode, true_mode, rtol = 0.01)
        @test isapprox(@inferred(bat_mode(posterior, MaxDensityNelderMead(), initial_mode = samples.samples)).mode, true_mode, rtol = 0.01)
    end


    @testset "MaxDensityLBFGS" begin
        # Result Optim.maximize with BFGS is not type-stable:
        @test isapprox(bat_mode(posterior, MaxDensityLBFGS()).mode, true_mode, rtol = 0.01)
    end
end
