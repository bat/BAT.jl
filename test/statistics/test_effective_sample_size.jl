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
    v = nestedview(ElasticArray{Float64, 2}(undef, 3, 0))
    push!(v, [0, 0, 0])
    for i in 1:(10^4 - 1)
        push!(v, clamp.(last(v) .+ randn(rng, 3), [0, 0, 0], [10, 5, 8]))
    end

    v1 = flatview(v)[1, :]

    @testset "BAT.bat_integrated_autocorr_len" begin
        context = BATContext()
        @test @inferred(bat_integrated_autocorr_len(v1, GeyerAutocorLen(), context)).result ≈ 52.2404651916953
        @test @inferred(bat_integrated_autocorr_len(v, GeyerAutocorLen(), context)).result ≈ [52.240465191695314, 17.04353447359818, 38.393838710754345]

        @test @inferred(bat_integrated_autocorr_len(v1, SokalAutocorLen(), context)).result ≈ 44.243392655975356
        @test @inferred(bat_integrated_autocorr_len(v, SokalAutocorLen(), context)).result ≈ [44.243392655975356, 16.794891919657566, 31.94870020972804]
    end

    @testset "repetition-weight-exact ESS" begin
        context = BATContext()
        # Integer weights are run-length repetition counts: the ESS of the
        # weight-compressed samples must be exactly the ESS of the
        # run-length-decoded ordered chain:
        rng2 = stblrng()
        n_runs = 500
        vals = nestedview(ElasticArray{Float64, 2}(undef, 2, 0))
        push!(vals, [0.0, 0.0])
        for _ in 1:(n_runs - 1)
            push!(vals, last(vals) .+ randn(rng2, 2))
        end
        weights = rand(rng2, 1:5, n_runs)
        smpls_rle = DensitySampleVector(v = vals, logd = zeros(n_runs), weight = weights)

        expanded = nestedview(flatview(vals)[:, inverse_rle(1:n_runs, weights)])
        ess_rle = bat_eff_sample_size(smpls_rle, EffSampleSizeFromAC(), context).result
        ess_expanded = bat_eff_sample_size(expanded, EffSampleSizeFromAC(), context).result
        @test ess_rle ≈ ess_expanded
    end
end
