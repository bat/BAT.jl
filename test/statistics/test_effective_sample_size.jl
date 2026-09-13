# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Statistics
using ArraysOfArrays, ElasticArrays, StatsBase
using StableRNGs

@testset "effective_sample_size" begin
    stblrng() = StableRNG(789990641)

    # Primitive MCMC of flat distribution between [0, 0, 0] and [10, 5, 8]:
    rng = stblrng()
    v = VectorOfSimilarVectors(ElasticArray{Float64, 2}(undef, 3, 0))
    push!(v, [0, 0, 0])
    for i in 1:(10^4 - 1)
        push!(v, clamp.(last(v) .+ randn(rng, 3), [0, 0, 0], [10, 5, 8]))
    end

    v1 = flatview(v)[1, :]

    context = BATContext()
    algorithm = EffSampleSizeFromAC()

    @testset "BAT.bat_integrated_autocorr_len" begin
        context = BATContext()
        @test @inferred(bat_integrated_autocorr_len(v1, GeyerAutocorLen(), context)).result ≈ 52.2404651916953
        @test @inferred(bat_integrated_autocorr_len(v, GeyerAutocorLen(), context)).result ≈ [52.240465191695314, 17.04353447359818, 38.393838710754345]

        @test @inferred(bat_integrated_autocorr_len(v1, SokalAutocorLen(), context)).result ≈ 44.243392655975356
        @test @inferred(bat_integrated_autocorr_len(v, SokalAutocorLen(), context)).result ≈ [44.243392655975356, 16.794891919657566, 31.94870020972804]
    end

    @testset "autocorrelation ESS" begin
        @test bat_eff_sample_size(fill(1.0, 8), algorithm, context).result == 8.0
        @test bat_eff_sample_size(repeat([-1.0, 1.0], 8), algorithm, context).result == 16.0
    end

    @testset "process provenance" begin
        id(chain, step) = BAT.MCMCSampleID(Int32(chain), Int32(1), Int32(1), Int64(step), Int32(1), true)
        values = [[0.0], [1.0], [0.0], [2.0], [3.0], [2.0]]
        info = [id(1, step) for step in 1:3]
        append!(info, [id(2, step) for step in 1:3])
        samples = DensitySampleVector(v = values, logd = zeros(6), info = info)
        shuffled = samples[[4, 1, 5, 2, 6, 3]]

        @test bat_eff_sample_size(shuffled, algorithm, context).result ≈
            bat_eff_sample_size(samples, algorithm, context).result

        scaled = DensitySampleVector(
            v = samples.v,
            logd = samples.logd,
            weight = fill(1e100, length(samples)),
            info = samples.info,
        )
        @test bat_eff_sample_size(scaled, algorithm, context).result ≈
            bat_eff_sample_size(samples, algorithm, context).result
    end

    @testset "pooled ESS" begin
        unequal = BAT._pooled_ess([[1000.0], [10.0]], [1.0, 1.0])
        @test unequal ≈ [1 / (0.25 / 1000 + 0.25 / 10)]

        masses = [3.0, 5.0]
        @test BAT._pooled_ess([[3.0], [5.0]], masses) ≈
            BAT._pooled_ess([[3.0], [5.0]], 1e300 .* masses)
    end
end
