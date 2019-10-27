# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ArraysOfArrays, ElasticArrays


struct _SampleInfo
    x::Int
end

_SampleInfo() = _SampleInfo(0)


@testset "density_sample" begin
    param1 = [Float64(1.),-0.1, 0.5]
    param2 = [Float64(-2.5), 0.2, 2.7]
    param4 = ones(Float64, 3)
    ds1 = @inferred PosteriorSample(param1, Float32(-3.3868156), Float32(-1.8932744), 1, _SampleInfo(7))
    ds2 = @inferred PosteriorSample(param2, Float32(-2.8723492), Float32(-7.2894223), 2, _SampleInfo(8))
    ds4 = @inferred PosteriorSample(param4, Float32(-4.2568156), Float32(-1.2892343), 4, _SampleInfo(9))
    
    @testset "PosteriorSample" begin
        @test typeof(ds1)  <: AbstractDensitySample
        @test typeof(ds1)  <: PosteriorSample{Float64,Float32,Int,_SampleInfo}

        @test nparams(ds1) == 3
        @test nparams.(ds1) == 3

        @test typeof(ds2)  <: PosteriorSample{Float64,Float32,Int,_SampleInfo}

        @test ds1 != ds4
        @test ds1 != ds2
        ds3 = @inferred PosteriorSample(param1, Float32(-3.3868156), Float32(-1.8932744), 1, _SampleInfo(7))
        @test ds1 == ds3
    end

    @testset "PosteriorSampleVector" begin
        dsv1 = @inferred PosteriorSampleVector{Float64,Float32,Int,_SampleInfo}(undef, 0, 3)
        @test typeof(dsv1) <: PosteriorSampleVector{Float64,Float32,Int,_SampleInfo}
        
        @test size(dsv1) == (0,)
        push!(dsv1, ds1)
        @test size(dsv1) == (1,)
        @test dsv1[1] == ds1
        @test IndexStyle(dsv1) == IndexLinear()

        push!(dsv1, ds2)
        dsv2 = @inferred PosteriorSampleVector{Float64,Float64,Float32,_SampleInfo}(undef, 0, 3)
        push!(dsv2, ds2)
        push!(dsv2, ds4)        
        append!(dsv1, dsv2)
        @test dsv1[4] == ds4
        @test dsv1[2] == ds2
    end
end
