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
end
