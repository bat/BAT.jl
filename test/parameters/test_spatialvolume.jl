# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using ArraysOfArrays, IntervalSets

@testset "SpatialVolume" begin
    lo = [-1., -0.1]
    hi = [2.,1]
    hyperRectVolume = BAT.HyperRectVolume(lo, hi)
    @testset "BAT.SpatialVolume" begin
        @test eltype(@inferred BAT.HyperRectVolume([-1., 0.5], [2.,1])) == Float64
        @test size(rand(MersenneTwister(7002), hyperRectVolume)) == (2,)
        @test size(rand(MersenneTwister(7002), hyperRectVolume, 3)) == (3,)
    end
    @testset "BAT.HyperRectVolume" begin
        @test typeof(@inferred BAT.HyperRectVolume([-1., 0.5], [2.,1])) <: SpatialVolume{Float64}
        @test_throws ArgumentError BAT.HyperRectVolume([-1.], [2.,1])

        @test [0.0, 0.0] in hyperRectVolume
        @test ([0.5, 2] in hyperRectVolume) == false

        @test ndims(hyperRectVolume) == 2

        res = @inferred rand!(MersenneTwister(7002), hyperRectVolume, zeros(2))
        @test typeof(res) <: AbstractArray{Float64, 1}
        @test size(res) == (2,)
        @test res in hyperRectVolume
        res = @inferred rand!(MersenneTwister(7002), hyperRectVolume, VectorOfSimilarVectors(zeros(2,3)))
        @test typeof(res) <: VectorOfSimilarVectors{Float64,Array{Float64,2}}
        @test size(res) == (3,)

        res = @inferred similar(hyperRectVolume)
        @test typeof(res) <: HyperRectVolume{Float64}

        a = @inferred HyperRectVolume([0.0,0.0],[1.0,1.0])
        b = @inferred HyperRectVolume([0.5,0.9],[0.8,1.5])
        res = @inferred intersect(a, b)
        @test res.lo ≈ [0.5, 0.9]
        @test res.hi ≈ [0.8, 1.0]
        @test isempty(res) == false

        a = @inferred HyperRectVolume([0.0,0.0],[0.3,0.4])
        res = @inferred intersect(a, b)
        @test res.lo ≈ [0.5, 0.9]
        @test res.hi ≈ [0.3, 0.4]
        @test isempty(res)
    end

    @testset "log_volume" begin
        @test log_volume(@inferred HyperRectVolume([-1.,0.],[1.,3.])) ≈
            1.79175946
        @test log_volume(@inferred HyperRectVolume([-0.0001],[0.0000])) ≈
            -9.210340371
    end

    @testset "fromuhc" begin
        @test inv(fromuhc!) == BAT.inv_fromuhc!
        @test inv(BAT.inv_fromuhc!) == fromuhc!
        @test inv(fromuhc) == BAT.inv_fromuhc
        @test inv(BAT.inv_fromuhc) == fromuhc

        y = zeros(Float64,2)
        x = similar(y)
        fromuhc!(x, y, hyperRectVolume)
        @test x ≈ hyperRectVolume.lo
        u = similar(y)
        inv(fromuhc!)(u, x, hyperRectVolume)
        @test u ≈ y
        y = ones(Float64, 2)
        res = @inferred fromuhc(y, hyperRectVolume)
        @test res ≈ hyperRectVolume.hi
        @test inv(fromuhc)(res, hyperRectVolume) ≈ y
        y = VectorOfSimilarVectors([0.2 0.5 0.5; 0.9 0.7 0.5])
        res = @inferred fromuhc(y, hyperRectVolume)
        @test res ≈ VectorOfSimilarVectors([-0.4 0.5 0.5; 0.89 0.67 0.45])
    end
end
