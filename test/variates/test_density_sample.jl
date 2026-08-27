# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, BATTestCases
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
        @test typeof(dsv1) <: DensitySampleVector{<:AbstractVector{Float64},Float32,Int,_SampleInfo,_SampleAux}
        
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
        low_weight_dsv = BAT.DensitySampleVector(v = dsv_merged.v, logd = dsv_merged.logd, weight = w2, info = dsv_merged.info, aux = dsv_merged.aux)
        num_samples_dropped = length(low_weight_dsv) - length(BAT.drop_low_weight_samples(low_weight_dsv, 0.05, threshold=0.1))
        @test @inferred(length(BAT.drop_low_weight_samples(low_weight_dsv, 0.05, threshold=0.1))) == 2
        @test BAT.drop_low_weight_samples(low_weight_dsv, 0.05, threshold=10^-6) == low_weight_dsv
        @test BAT.drop_low_weight_samples(low_weight_dsv, 10^-6, threshold=0.1) == low_weight_dsv

        dsv_similar = @inferred(similar(dsv_merged))
        for v in dsv_similar.v
            @test isassigned(v) == false
        end

        gs = GaussianShell(n=5)
        x1 = rand(5)
        x2 = rand(5)
        v_gs = convert(ArrayOfSimilarArrays, [x1, x2])
        logd_gs = [logpdf(gs, x1), logpdf(gs, x2)]

        dsv_gs1 = DensitySampleVector(v = v_gs, logd = logd_gs, weight = [1,1])
        dsv_gs2 = DensitySampleVector(v = v_gs, logd = logd_gs, weight = :multiplicity)

        @test dsv_gs1 == dsv_gs2
        @test dsv_gs1.v == v_gs
        @test dsv_gs2.v == v_gs
        @test dsv_gs1.weight == [1,1]
        @test dsv_gs2.weight == [1,1]
        
        @test @inferred(length(DensitySampleVector(v = dsv_merged.v, logd = dsv_merged.logd, weight = :multiplicity))) == @inferred(length(dsv_merged))-1

        # An empty sample vector stays empty under multiplicity weighting:
        dsv_empty = DensitySampleVector(v = dsv_merged.v[1:0], logd = dsv_merged.logd[1:0], weight = :multiplicity)
        @test isempty(dsv_empty) && isempty(dsv_empty.v) && isempty(dsv_empty.weight)

        rtol = eps(typeof(float(1)))
        X = @inferred(flatview(dsv_merged.v))
        w = @inferred(ProbabilityWeights(dsv_merged.weight))
        rows = eachrow(X)

        dsv_mean = @inferred(mean(dsv_merged))
        @test @inferred(length(rows)) == @inferred(length(dsv_mean))
        for i in eachindex(dsv_mean)
            @test isapprox(@inferred(mean(collect(rows)[i], w)), dsv_mean[i], rtol=rtol)
        end
        
        dsv_std = @inferred(std(dsv_merged))
        @test @inferred(length(rows)) == @inferred(length(dsv_std))
        for i in eachindex(dsv_std)
            # Uncorrected empirical-measure moments (weight provenance is
            # deliberately erased at the sample-vector level):
            @test isapprox(@inferred(std(collect(rows)[i], w, corrected=false)), dsv_std[i], rtol=rtol)
        end

        dsv_med = @inferred(median(dsv_merged))
        @test @inferred(length(rows)) == @inferred(length(dsv_med))
        @test dsv_med == [1.0, 1.0, 1.0]

        @testset "weighted quantiles use empirical mass" begin
            values = [-3.0, -1.0, 2.0, 8.0]
            sample_weights = [0, 1, 2, 5]
            probabilities = [0.0, 0.25, 0.5, 0.75, 1.0]
            expected = [-1.0, 2.0, 8.0, 8.0, 8.0]

            compressed = DensitySampleVector(v = values, logd = zeros(4), weight = sample_weights)
            expanded = DensitySampleVector(
                v = [-1.0, 2.0, 2.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                logd = zeros(8),
            )

            @test quantile.(Ref(compressed), probabilities) == expected
            @test quantile.(Ref(expanded), probabilities) == expected

            permutations = (
                collect(p) for p in Iterators.product(ntuple(_ -> eachindex(values), length(values))...)
                if allunique(p)
            )
            for permutation in permutations
                permuted = DensitySampleVector(
                    v = values[permutation],
                    logd = zeros(4),
                    weight = sample_weights[permutation],
                )
                @test quantile.(Ref(permuted), probabilities) == expected
            end

            for scale in (nextfloat(0.0), 0.5, 2.0, floatmax(Float64) / 8)
                scaled = DensitySampleVector(
                    v = values,
                    logd = zeros(4),
                    weight = scale .* sample_weights,
                )
                @test quantile.(Ref(scaled), probabilities) == expected
            end

            tied = DensitySampleVector(v = [0.0, 0.0, 1.0], logd = zeros(3), weight = [1, 2, 1])
            @test quantile.(Ref(tied), probabilities) == [0.0, 0.0, 0.0, 0.0, 1.0]

            endpoint_weights = [typemax(Int), typemax(Int) - 1, 1, 2]
            endpoints = DensitySampleVector(v = values, logd = zeros(4), weight = endpoint_weights)
            @test quantile(endpoints, 0.0) == -3.0
            @test quantile(endpoints, 1.0) == 8.0

            samples32 = DensitySampleVector(
                v = Float32[-1, 1],
                logd = zeros(Float32, 2),
                weight = Float32[1, 2],
            )
            samplesbig = DensitySampleVector(
                v = BigFloat[-1, 1],
                logd = zeros(BigFloat, 2),
                weight = BigFloat[1, 2],
            )
            @test @inferred(quantile(samples32, 0.5f0)) === 1.0f0
            quantile_big = @inferred quantile(samplesbig, big"0.5")
            @test quantile_big == big"1.0"
            @test quantile_big isa BigFloat

            vector_values = convert(ArrayOfSimilarArrays, [Float32[-1, 0], Float32[1, 2]])
            vector_samples = DensitySampleVector(
                v = vector_values,
                logd = zeros(Float32, 2),
                weight = Float32[1, 2],
            )
            vector_median = @inferred quantile(vector_samples, 0.5f0)
            @test vector_median == Float32[1, 2]
            @test eltype(vector_median) === Float32

            nan_samples = DensitySampleVector(v = [0.0, NaN], logd = zeros(2), weight = [1, 1])
            empty_samples = DensitySampleVector(v = Float64[], logd = Float64[], weight = Float64[])
            zero_weight_samples = DensitySampleVector(v = [0.0, 1.0], logd = zeros(2), weight = [0, 0])
            negative_weight_samples = DensitySampleVector(v = [0.0, 1.0], logd = zeros(2), weight = [1, -1])
            @test isnan(quantile(nan_samples, 0.5))
            @test_throws ArgumentError quantile(empty_samples, 0.5)
            @test_throws ArgumentError quantile(zero_weight_samples, 0.5)
            @test_throws ArgumentError quantile(negative_weight_samples, 0.5)
            @test_throws ArgumentError quantile(compressed, -0.1)
            @test_throws ArgumentError quantile(compressed, 1.1)
        end

        dsv_mode = @inferred(mode(dsv_merged))
        for i in eachindex(dsv_mode)
            @test @inferred(isapprox(dsv_mode[i], mode(collect(rows)[i]), rtol=rtol))
        end

        @test @inferred(isapprox(@inferred(cor(X, w, 2)), @inferred(cor(dsv_merged)), rtol=rtol))

        @testset "weighted statistics ignore global weight scale" begin
            values = convert(ArrayOfSimilarArrays, [[0.0, 0.0], [2.0, 4.0]])
            for weights in ([typemax(Int), typemax(Int)], [1e308, 1e308], [1e-320, 1e-320])
                samples = DensitySampleVector(v = values, logd = zeros(2), weight = weights)
                @test mean(samples) ≈ [1.0, 2.0]
                @test var(samples) ≈ [1.0, 4.0]
                @test std(samples) ≈ [1.0, 2.0]
                @test quantile(samples, 0.5) ≈ [0.0, 0.0]
                @test cov(samples) ≈ [1.0 2.0; 2.0 4.0]
                @test cor(samples) ≈ ones(2, 2)
            end
        end
    end
end
