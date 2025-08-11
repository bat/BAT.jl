# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, PDMats
using DensityInterface

@testset "bat_dist_measure" begin
    context = BATContext()

    mvt = @inferred MvTDist(1.5, PDMat([2.0 1.0; 1.0 3.0]))
    mvdd = @inferred BAT.BATDistMeasure(mvt)

    @testset "properties" begin
        @test parent(mvdd) == mvt
    end

    @testset "logdensityof" begin
        @test (@inferred logdensityof(mvdd, [0.0, 0.0])) ≈ -2.64259602
    end

    @testset "statistics" begin
        mvn = @inferred(product_distribution([Normal(-1.0), Normal(0.0), Normal(1.0)]))
        dist_density = @inferred(BAT.BATDistMeasure(mvn))
        @test @inferred(bat_findmode(dist_density, context)).result == mode(mvn)
    end
end
