# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Distributions, StatsBase, ValueShapes

@testset "bat_findmode" begin
    prior = NamedTupleDist(
        x = Normal(2.0, 1.0),
        c = [4, 5],
        a = MvNormal([1.5, 0.5, 2.5], Matrix{Float32}(I, 3, 3))
    )

    posterior = PosteriorDensity(v -> LogDVal(0), prior)

    true_mode = [2.0, 1.5, 0.5, 2.5]

    samples = @inferred(bat_sample(prior, 10^5)).result


    @testset "ModeAsDefined" begin
        @test @inferred(BAT.default_mode_estimator(prior)) == ModeAsDefined()
        @test @inferred(bat_findmode(prior)).result == true_mode
        @test @inferred(bat_findmode(BAT.DistributionDensity(prior), ModeAsDefined())).result == true_mode
    end


    @testset "MaxDensitySampleSearch" begin
        @test @inferred(BAT.default_mode_estimator(samples)) == MaxDensitySampleSearch()
        @test @inferred(bat_findmode(samples, MaxDensitySampleSearch())) == @inferred(bat_findmode(samples))
        m = bat_findmode(samples, MaxDensitySampleSearch())
        @test samples[m.mode_idx].v == m.result
        @test isapprox(m.result, true_mode, rtol = 0.05)
    end


    @testset "MaxDensityNelderMead" begin
        @test @inferred(BAT.default_mode_estimator(posterior)) == MaxDensityNelderMead()
        @test isapprox(@inferred(bat_findmode(posterior, MaxDensityNelderMead())).result, true_mode, rtol = 0.01)
        @test isapprox(@inferred(bat_findmode(posterior, MaxDensityNelderMead(), initial_mode = rand(prior))).result, true_mode, rtol = 0.01)
        @test isapprox(@inferred(bat_findmode(posterior, MaxDensityNelderMead(), initial_mode = samples)).result, true_mode, rtol = 0.01)
    end


    @testset "MaxDensityLBFGS" begin
        # Result Optim.maximize with BFGS is not type-stable:
        @test isapprox(bat_findmode(posterior, MaxDensityLBFGS()).result, true_mode, rtol = 0.01)
    end
end
