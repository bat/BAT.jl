# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using ArraysOfArrays, ElasticArrays


struct _SampleInfo
    x::Int
end

struct _SampleAux
    x::Float32
end

_SampleInfo() = _SampleInfo(0)
_SampleAux() = _SampleInfo(0)


@testset "density_sample" begin
    param1 = [Float64(1.),-0.1, 0.5]
    param2 = [Float64(-2.5), 0.2, 2.7]
    param4 = ones(Float64, 3)
    ds1 = @inferred DensitySample(param1, Float32(-3.3868156), 1, _SampleInfo(7), _SampleAux(0.378f0))
    ds2 = @inferred DensitySample(param2, Float32(-2.8723492), 2, _SampleInfo(8), _SampleAux(0.435f0))
    ds4 = @inferred DensitySample(param4, Float32(-4.2568156), 4, _SampleInfo(9), _SampleAux(0.612f0))
    
    @testset "DensitySample" begin
        @test typeof(ds1)  <: AbstractDensitySample
        @test typeof(ds1)  <: DensitySample{Float64,Float32,Int,_SampleInfo}

        @test nparams(ds1) == 3
        @test nparams.(ds1) == 3

        @test typeof(ds2)  <: DensitySample{Float64,Float32,Int,_SampleInfo,_SampleAux}

        @test ds1 != ds4
        @test ds1 != ds2
        ds3 = @inferred DensitySample(param1, Float32(-3.3868156), 1, _SampleInfo(7), _SampleAux(0.378f0))
        @test ds1 == ds3
    end

    @testset "DensitySampleVector" begin
        dsv1 = @inferred DensitySampleVector{Float64,Float32,Int,_SampleInfo,_SampleAux}(undef, 0, 3)
        @test typeof(dsv1) <: DensitySampleVector{Float64,Float32,Int,_SampleInfo,_SampleAux}
        
        @test size(dsv1) == (0,)
        push!(dsv1, ds1)
        @test size(dsv1) == (1,)
        @test dsv1[1] == ds1
        @test IndexStyle(dsv1) == IndexLinear()

        push!(dsv1, ds2)
        dsv2 = @inferred DensitySampleVector{Float64,Float64,Float32,_SampleInfo,_SampleAux}(undef, 0, 3)
        push!(dsv2, ds2)
        push!(dsv2, ds4)        
        append!(dsv1, dsv2)
        @test dsv1[4] == ds4
        @test dsv1[2] == ds2
    end
end
