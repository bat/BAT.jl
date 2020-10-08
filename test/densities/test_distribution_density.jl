# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ArraysOfArrays, Distributions, StatsBase, PDMats, IntervalSets, ValueShapes

@testset "distribution_density" begin
    mvt = @inferred MvTDist(1.5, PDMat([2.0 1.0; 1.0 3.0]))
    mvdd = @inferred BAT.DistributionDensity(mvt)

    @testset "properties" begin
        @test typeof(mvdd) <: AbstractDensity
        @test parent(mvdd) == mvt
        @test totalndof(mvdd) == 2
    end

    @testset "BAT.eval_logval_unchecked" begin
        @test (@inferred BAT.eval_logval_unchecked(mvdd, [0.0, 0.0])) ≈ -2.64259602
    end

    @testset "BAT.var_bounds" begin
        let
            dist = @inferred NamedTupleDist(a = 5, b = Normal(), c = -4..5, d = MvNormal([1.2 0.5; 0.5 2.1]), e = [Normal(1.1, 0.2)] )
            density = @inferred BAT.DistributionDensity(dist)
            s = sampler(density)
            @test all([rand(s) in BAT.var_bounds(density) for i in 1:10^4])
        end
    end

    @testset "histogram support" begin
        d1 = Normal(1, 2)
        X1 = rand(d1, 10^5)
        h1 = fit(Histogram, X1)
        dd1 = @inferred BAT.DistributionDensity(h1)
        s1 = @inferred sampler(dd1)
        @test all([rand(s1) in BAT.var_bounds(dd1) for i in 1:10^4])

        d2 = MvNormal([1.0 1.5; 1.5 4.0])
        X2 = rand(d2, 10^5)
        h2 = fit(Histogram, (X2[1,:], X2[2,:]))
        dd2 = @inferred BAT.DistributionDensity(h2)
        s2 = @inferred sampler(dd2)
        @test all([rand(s2) in BAT.var_bounds(dd2) for i in 1:10^4])

        ntd = NamedTupleDist(a = h1, b = h2)
        ntdd = @inferred BAT.DistributionDensity(ntd)
        snt = @inferred sampler(ntdd)
        @test all([rand(snt) in BAT.var_bounds(ntdd) for i in 1:10^4])
    end
end
