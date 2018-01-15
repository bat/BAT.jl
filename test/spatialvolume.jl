# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

using IntervalSets

@testset "SpatialVolume" begin
    lo = [-1., -0.1]
    hi = [2.,1]
    hyperRectVolume = BAT.HyperRectVolume(lo, hi)
    @testset "BAT.SpatialVolume" begin
        @test eltype(@inferred BAT.HyperRectVolume([-1., 0.5], [2.,1])) == Float64
        @test size(rand(MersenneTwister(7002), hyperRectVolume)) == (2,)
        @test size(rand(MersenneTwister(7002), hyperRectVolume, 3)) == (2, 3,)
    end
    @testset "BAT.HyperRectVolume" begin
        @test typeof(@inferred BAT.HyperRectVolume([-1., 0.5], [2.,1])) <: SpatialVolume{Float64}
        @test_throws ArgumentError BAT.HyperRectVolume([-1.], [2.,1])

        @test [0.0, 0.0] in hyperRectVolume
        @test ([0.5, 2] in hyperRectVolume) == false

        @test ndims(hyperRectVolume) == 2

        res = rand!(MersenneTwister(7002), hyperRectVolume, zeros(2))
        @test typeof(res) <: AbstractArray{Float64, 1}
        @test size(res) == (2,)
        @test res in hyperRectVolume
        res = rand!(MersenneTwister(7002), hyperRectVolume, zeros(2,3))
        @test typeof(res) <: AbstractArray{Float64, 2}
        @test size(res) == (2, 3)
    end

    @testset "log_volume" begin
        @test log_volume(@inferred HyperRectVolume([-1.,0.],[1.,3.])) ≈
            1.79175946
        @test log_volume(@inferred HyperRectVolume([-0.0001],[0.0000])) ≈
            -9.210340371
    end
end
