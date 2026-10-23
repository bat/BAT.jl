# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT, BATTestCases
using Distributions
using StatsBase
using Test

using ArraysOfArrays, ElasticArrays, FillArrays, StructArrays, ValueShapes
import TypedTables
import MeasureBase
using MeasureBase: productmeasure, asmeasure, StdUniform, StdNormal, logdensities


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
        @test typeof(ds1) <: DensitySample{Vector{Float64},Float32,Int,_SampleInfo}
        @test typeof(ds2) <: DensitySample{Vector{Float64},Float32,Int,_SampleInfo,_SampleAux}
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

        shaped_dsv1 = @test_deprecated shape.(dsv1)
        @test shaped_dsv1 isa DensitySampleVector
        @test shaped_dsv1.v isa StructVector && varshape(shaped_dsv1) == shape
        @test shaped_dsv1[2].v == (x = ds2.v[1], y = ds2.v[2:3])
        @test @inferred(broadcast(unshaped, shaped_dsv1)).v == dsv1.v
        @test flatview(unshaped.(dsv1).v) === flatview(dsv1.v)

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
        @test isempty(dsv_empty)

        rtol = eps(float(eltype(dsv_merged.weight)))
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

            samples16 = DensitySampleVector(
                v = [1.0, 2.0], logd = zeros(2), weight = Float16[1, 3],
            )
            @test quantile(samples16, 0.25) == 1.0

            zero_mass_nan = DensitySampleVector(v = [1.0, NaN], logd = zeros(2), weight = [1, 0])
            @test quantile.(Ref(zero_mass_nan), probabilities) == ones(5)

            @test quantile.(Ref(compressed), probabilities) == quantile.(Ref(expanded), probabilities) == expected
        end

        dsv_mode = @inferred(mode(dsv_merged))
        for i in eachindex(dsv_mode)
            @test @inferred(isapprox(dsv_mode[i], mode(collect(rows)[i]), rtol=rtol))
        end

        @test @inferred(isapprox(@inferred(cor(X, w, 2)), @inferred(cor(dsv_merged)), rtol=rtol))

        @testset "marginal histograms" begin
            smpls = DensitySampleVector(v = nestedview([0.0 0.5 1.0 2.0; 1.0 1.0 1.0 1.0]), logd = zeros(4), weight = [1, 2, 1, 3])
            hists_coarse = BAT._marginal_histograms(smpls, nbins = 2)
            hists_fine = BAT._marginal_histograms(smpls, nbins = 10)
            @test all(h -> sum(h.weights) == sum(smpls.weight), vcat(hists_coarse, hists_fine))
            @test length(hists_fine[1].weights) > length(hists_coarse[1].weights)
        end

        @testset "growing requires flat storage" begin
            μ = productmeasure((a = StdNormal(), b = StdNormal()^2))
            smpls = DensitySampleVector(rand(μ^3), μ)
            @test_throws ArgumentError push!(deepcopy(smpls), smpls[1])
            @test_throws ArgumentError append!(deepcopy(smpls), smpls)
        end

        @testset "weighted statistics ignore global weight scale" begin
            values = [0.0, 2.0]
            unit = DensitySampleVector(v = values, logd = zeros(2), weight = [1, 1])
            scaled = DensitySampleVector(
                v = values, logd = zeros(2), weight = [typemax(Int), typemax(Int)]
            )
            @test mean(scaled) ≈ mean(unit)
        end
    end

    @testset "structured samples" begin
        μ = productmeasure((
            a = asmeasure(Normal()),
            b = asmeasure(MvNormal([2.0 0.5; 0.5 1.0])),
            c = MeasureBase.Dirac(4.2),
            d = productmeasure((x = StdUniform(), y = StdNormal()^(2, 3))),
        ))
        n = 50
        X = rand(μ^n)
        weight = rand(1:3, n)
        smpls = DensitySampleVector(X, μ, logd = logdensities(μ, X), weight = weight)
        @test smpls.v.c isa Fill
        @test varshape(smpls) == varshape(μ)
        @test occursin("varshape", sprint(show, smpls))
        @test sprint(show, MIME"text/plain"(), smpls) isa String

        flat = unshaped.(smpls)
        @test flat.v isa VectorOfSimilarVectors
        @test flatview(flat.v) == BAT.flatbatch(smpls.v)
        @test DensitySampleVector(flatview(flat.v), μ, logd = smpls.logd, weight = weight) == smpls

        nts = [(a = x.a, b = Vector(x.b), c = x.c, d = (x = x.d.x, y = Matrix(x.d.y))) for x in X]
        smpls_nt = DensitySampleVector(v = nts, logd = smpls.logd, weight = weight)
        @test smpls_nt.v.b isa ArrayOfSimilarArrays && smpls_nt.v.d.y isa ArrayOfSimilarArrays
        @test DensitySampleVector(nts, μ, logd = smpls.logd, weight = weight) == smpls

        m, m_flat = mean(smpls), mean(flat)
        @test m isa NamedTuple && m.c == 4.2
        @test vcat(m.a, m.b, m.d.x, vec(m.d.y)) ≈ m_flat
        v, v_flat = var(smpls), var(flat)
        @test v.c == 0 && vcat(v.a, v.b, v.d.x, vec(v.d.y)) ≈ v_flat
        @test std(smpls).d.y ≈ sqrt.(v.d.y)
        for (f, f_flat) in ((s -> quantile(s, 0.3), s -> quantile(s, 0.3)), (minimum, minimum), (maximum, maximum), (mode, mode))
            r = f(smpls)
            @test r.c == 4.2
            @test vcat(r.a, r.b, r.d.x, vec(r.d.y)) == f_flat(flat)
        end
        @test minimum(flat) == minimum(flatview(flat.v), dims = 2)[:]
        @test cov(smpls) ≈ cov(flat) && cor(smpls) ≈ cor(flat)

        @test smpls[3].v.c == 4.2 && smpls[3].v.b == smpls.v.b[3]
        sub = smpls[2:5]
        @test sub.v.c isa Fill && length(sub) == 4 && sub[1] == smpls[2]
        both = vcat(smpls, sub)
        @test both.v.c isa Fill && length(both) == n + 4 && both[n + 1] == smpls[2]
        @test merge(smpls, sub) == both
        @test isapprox(smpls, deepcopy(smpls)) && !isapprox(smpls, flat)
    end
end
