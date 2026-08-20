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
        # When the caller knows the weights are run-length repetition
        # counts (only the MCMC layer knows - the generic sample-vector
        # level deliberately erases weight provenance), the ESS of the
        # weight-compressed samples is exactly the ESS of the
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
        ess_rle = BAT._repetition_exact_ess(smpls_rle, EffSampleSizeFromAC(), context)
        ess_expanded = bat_eff_sample_size(expanded, EffSampleSizeFromAC(), context).result
        @test ess_rle ≈ ess_expanded

        # Uniform repetition counts greater than one also decode:
        smpls_unif = DensitySampleVector(v = vals, logd = zeros(n_runs), weight = fill(2, n_runs))
        expanded_unif = nestedview(flatview(vals)[:, inverse_rle(1:n_runs, fill(2, n_runs))])
        @test BAT._repetition_exact_ess(smpls_unif, EffSampleSizeFromAC(), context) ≈
            bat_eff_sample_size(expanded_unif, EffSampleSizeFromAC(), context).result

        # The generic path stays provenance-neutral and scale-invariant:
        smpls_w = DensitySampleVector(v = vals, logd = zeros(n_runs), weight = float.(weights))
        smpls_w100 = DensitySampleVector(v = vals, logd = zeros(n_runs), weight = 100 .* float.(weights))
        @test bat_eff_sample_size(smpls_w, EffSampleSizeFromAC(), context).result ≈
            bat_eff_sample_size(smpls_w100, EffSampleSizeFromAC(), context).result
    end
end
