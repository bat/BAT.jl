# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Distributions, StatsBase, ValueShapes

@testset "mode_estimators" begin
    prior = NamedTupleDist(
        x = Normal(2.0, 1.0),
        c = [4, 5],
        a = MvNormal([1.5, 0.5, 2.5], Matrix{Float32}(I, 3, 3))
    )

    posterior = PosteriorDensity(LogDVal(0), prior)

    true_mode_flat = [2.0, 1.5, 0.5, 2.5]
    true_mode = stripscalar(varshape(prior)(true_mode_flat))

    samples = @inferred(bat_sample(prior, IIDSampling(nsamples = 10^5))).result


    function test_findmode(posterior, algorithm, rtol)
        res = @inferred(bat_findmode(posterior, algorithm))
        @test keys(stripscalar(res.result)) == keys(true_mode)
        @test isapprox(unshaped(res.result), true_mode_flat, rtol = rtol)
    end

    function test_findmode_noinferred(posterior, algorithm, rtol)
        res = (bat_findmode(posterior, algorithm))
        @test keys(stripscalar(res.result)) == keys(true_mode)
        @test isapprox(unshaped(res.result), true_mode_flat, rtol = rtol)
    end


    @testset "ModeAsDefined" begin
        @test @inferred(bat_findmode(prior, ModeAsDefined())).result[] == true_mode
        @test @inferred(bat_findmode(BAT.DistributionDensity(prior), ModeAsDefined())).result[] == true_mode
        let post_modes = @inferred(bat_findmode(posterior)).result[]
            for k in keys(post_modes)
                @test isapprox(post_modes[k], true_mode[k], atol=0.001)
            end
        end
    end


    @testset "MaxDensitySampleSearch" begin
        @test @inferred(bat_findmode(samples, MaxDensitySampleSearch())).result isa ShapedAsNT
        m = bat_findmode(samples, MaxDensitySampleSearch())
        @test samples[m.mode_idx].v == stripscalar(m.result)
        @test isapprox(unshaped(m.result), true_mode_flat, rtol = 0.05)
    end


    @testset "MaxDensityNelderMead" begin
        test_findmode(posterior,  MaxDensityNelderMead(), 0.01)
    end


    @testset "MaxDensityLBFGS" begin
        # Result Optim.maximize with LBFGS is not type-stable:
        test_findmode_noinferred(posterior,  MaxDensityLBFGS(), 0.01)
    end
end
