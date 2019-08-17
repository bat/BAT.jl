# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ArraysOfArrays, Distributions, PDMats

@testset "mvdist_density" begin
    mvt = @inferred MvTDist(1.5, PDMat([2.0 1.0; 1.0 3.0]))
    mvdd = @inferred DistributionDensity(mvt)

    @testset "mvdist_density" begin
        @test typeof(mvdd) <: AbstractDensity
        @test parent(mvdd) == mvt
        @test nparams(mvdd) == 2
    end

    @testset "density_logval" begin
        @test (@inferred density_logval(mvdd, [0.0, 0.0])) ≈ -2.64259602
    end
end
