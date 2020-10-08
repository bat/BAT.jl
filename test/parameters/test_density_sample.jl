# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Distributions
using StatsBase
using Test

using ArraysOfArrays, ElasticArrays, ValueShapes
import TypedTables


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
    
    @inferred(Base.Broadcast.broadcastable(ds1)) isa Ref && Base.Broadcast.broadcastable(ds1).x == Ref(ds1).x

    naive_ds = @inferred DensitySample([1.0, 2.0, 3.0], 4.0, 5.0, nothing, nothing)
    similar_ds = @inferred(similar(naive_ds))
    @test isnan.(similar_ds.v) == @inferred(ones(Int, @inferred(length(similar_ds.v))))

    @testset "DensitySample" begin
        @test typeof(ds1)  <: DensitySample{Vector{Float64},Float32,Int,_SampleInfo}

        @test typeof(ds2)  <: DensitySample{Vector{Float64},Float32,Int,_SampleInfo,_SampleAux}

        @test ds1 != ds4
        @test ds1 != ds2
        ds3 = @inferred DensitySample(param1, Float32(-3.3868156), 1, _SampleInfo(7), _SampleAux(0.378f0))
        @test ds1 == ds3
    end

    @testset "DensitySampleVector" begin
        dsv1 = @inferred DensitySampleVector{Vector{Float64},Float32,Int,_SampleInfo,_SampleAux}(undef, 0, 3)
        @test typeof(dsv1) <: DensitySampleVector{Vector{Float64},Float32,Int,_SampleInfo,_SampleAux}
        
        @test size(dsv1) == (0,)
        push!(dsv1, ds1)
        @test size(dsv1) == (1,)
        @test dsv1[1] == ds1
        @test IndexStyle(dsv1) == IndexLinear()

        push!(dsv1, ds2)
        dsv2 = @inferred DensitySampleVector{Vector{Float64},Float64,Float32,_SampleInfo,_SampleAux}(undef, 0, 3)
        push!(dsv2, ds2)
        push!(dsv2, ds4)        
        append!(dsv1, dsv2)
        @test dsv1[4] == ds4
        @test dsv1[2] == ds2

        shape = NamedTupleShape(x = ScalarShape{Real}(), y = ArrayShape{Real}(2)) 

        @test @inferred(broadcast(shape, dsv1)) isa DensitySampleVector
        @test broadcast(shape, dsv1).v isa ShapedAsNTArray
        @test @inferred(broadcast(unshaped, TypedTables.Table(broadcast(shape, dsv1)).v)) === dsv1.v

        @test shape.(dsv1)[1] == shape(dsv1[1])

        dsv_merged = @inferred(merge(dsv1, dsv2))
        @test vcat(dsv1, dsv2) == dsv_merged
        @test getindex(dsv_merged, 1:length(dsv_merged)) == dsv_merged
        @test getindex(dsv_merged, 1:length(dsv1)) == getindex(dsv1, 1:length(dsv1))

        w1 = 1:6
        w2 = append!(collect(1:5),10000)
        low_weight_dsv = BAT.DensitySampleVector(dsv_merged.v, dsv_merged.logd, weight=w2, info=dsv_merged.info, aux=dsv_merged.aux)
        num_samples_dropped = length(low_weight_dsv) - length(BAT.drop_low_weight_samples(low_weight_dsv, 0.05, threshold=0.1))
        @test @inferred(length(BAT.drop_low_weight_samples(low_weight_dsv, 0.05, threshold=0.1))) == 2
        @test BAT.drop_low_weight_samples(low_weight_dsv, 0.05, threshold=10^-6) == low_weight_dsv
        @test BAT.drop_low_weight_samples(low_weight_dsv, 10^-6, threshold=0.1) == low_weight_dsv

        dsv_similar = @inferred(similar(dsv_merged))
        for v in dsv_similar.v
            @test isassigned(v) == false
        end

        gs = BAT.GaussianShell(n=5)
        x1 = rand(5)
        x2 = rand(5)
        v_gs = ArrayOfSimilarArrays([x1, x2])
        logd_gs = [logpdf(gs, x1), logpdf(gs, x2)]

        dsv_gs1 = DensitySampleVector(v_gs, logd_gs, weight=[1,1])
        dsv_gs2 = DensitySampleVector(v_gs, logd_gs, weight=:RepetitionWeight)

        @test dsv_gs1 == dsv_gs2
        @test dsv_gs1.v == v_gs
        @test dsv_gs2.v == v_gs
        @test dsv_gs1.weight == [1,1]
        @test dsv_gs2.weight == [1,1]
        
        @test @inferred(length(DensitySampleVector(dsv_merged.v, dsv_merged.logd, weight=:RepetitionWeight))) == @inferred(length(dsv_merged))-1

        rtol = eps(typeof(float(1)))
        X = @inferred(flatview(dsv_merged.v))
        w = @inferred(FrequencyWeights(dsv_merged.weight))
        rows = eachrow(X)

        dsv_mean = @inferred(mean(dsv_merged))
        @test @inferred(length(rows)) == @inferred(length(dsv_mean))
        for i in eachindex(dsv_mean)
            @test isapprox(@inferred(mean(collect(rows)[i], w)), dsv_mean[i], rtol=rtol)
        end
        
        dsv_std = @inferred(std(dsv_merged))
        @test @inferred(length(rows)) == @inferred(length(dsv_std))
        for i in eachindex(dsv_std)
            @test isapprox(@inferred(std(collect(rows)[i], w, corrected=true)), dsv_std[i], rtol=rtol)
        end

        dsv_med = @inferred(median(dsv_merged))
        @test @inferred(length(rows)) == @inferred(length(dsv_med))
        for i in eachindex(dsv_med)
            @test isapprox(@inferred(median(collect(rows)[i], w)), dsv_med[i], rtol=rtol)
        end

        dsv_mode = @inferred(mode(dsv_merged))
        for i in eachindex(dsv_mode)
            @test @inferred(isapprox(dsv_mode[i], mode(collect(rows)[i]), rtol=rtol))
        end

        @test @inferred(isapprox(@inferred(cor(X, w, 2)), @inferred(cor(dsv_merged)), rtol=rtol))
    end
end
