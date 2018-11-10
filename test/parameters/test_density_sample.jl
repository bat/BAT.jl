# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ArraysOfArrays, ElasticArrays

@testset "density_sample" begin
    
    param1 = [Float64(1.),-0.1, 0.5]
    param2 = [Float64(-2.5), 0.2, 2.7]
    param4 = ones(Float64, 3)
    ds1 = @inferred DensitySample(param1, Float32(-3.3868156), Float32(-1.8932744), 1)
    ds2 = @inferred DensitySample(param2, Float32(-2.8723492), Float32(-7.2894223), 2)
    ds4 = @inferred DensitySample(param4, Float32(-4.2568156), Float32(-1.2892343), 4)
    
    @testset "DensitySample" begin
        @test typeof(ds1)  <: AbstractDensitySample
        @test typeof(ds1)  <: DensitySample{Float64,Float32,Int}

        @test length(ds1) == 3
        @test nparams(ds1) == 3

        @test typeof(ds2)  <: DensitySample{Float64,Float32,Int}

        @test ds1 != ds4
        @test ds1 != ds2
        ds3 = @inferred DensitySample(param1, Float32(-3.3868156), Float32(-1.8932744), 1)
        @test ds1 == ds3
    end

    @testset "DensitySampleVector" begin
        dsv1 = @inferred DensitySampleVector{Float64,Float32,Int}(3)
        @test typeof(dsv1) <: BAT.BATDataVector{DensitySample{Float64,Float32,Int,Vector{Float64}}}
        @test typeof(dsv1) <: DensitySampleVector{Float64,Float32,Int,VectorOfSimilarVectors{Float64,ElasticArray{Float64,2,1}},Vector{Float32},Vector{Int}}
        
        @test size(dsv1) == (0,)
        push!(dsv1, ds1)
        @test size(dsv1) == (1,)
        @test dsv1[1] == ds1
        @test IndexStyle(dsv1) == IndexLinear()

        push!(dsv1, ds2)
        dsv2 = @inferred DensitySampleVector{Float64, Float64, Float32}(3)
        push!(dsv2, ds2)
        push!(dsv2, ds4)        
        append!(dsv1, dsv2)
        @test dsv1[4] == ds4
        @test dsv1[2] == ds2
    end
end
