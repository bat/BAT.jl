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

        # The provenance-free heuristic path stays scale-invariant:
        smpls_w = DensitySampleVector(v = vals, logd = zeros(n_runs), weight = float.(weights))
        smpls_w100 = DensitySampleVector(v = vals, logd = zeros(n_runs), weight = 100 .* float.(weights))
        @test bat_eff_sample_size(smpls_w, EffSampleSizeFromAC(), context).result ≈
            bat_eff_sample_size(smpls_w100, EffSampleSizeFromAC(), context).result
    end

    @testset "process provenance ESS" begin
        context = BATContext()
        rng3 = stblrng()
        # Two independent walker chains with repetition weights and MCMC
        # sample ids, merged and shuffled - the per-sample provenance
        # reconstructs the exact ordered sequences:
        function mk_walker(rng, chainid, n)
            vals_w = nestedview(ElasticArray{Float64, 2}(undef, 2, 0))
            push!(vals_w, [0.0, 0.0])
            for _ in 1:(n - 1)
                push!(vals_w, last(vals_w) .+ randn(rng, 2))
            end
            w = rand(rng, 1:4, n)
            ids = [BAT.MCMCSampleID(Int32(chainid), Int32(1), Int32(1), Int64(i), Int32(1), true) for i in 1:n]
            DensitySampleVector(v = vals_w, logd = zeros(n), weight = w, info = ids)
        end
        w1 = mk_walker(rng3, 1, 300)
        w2 = mk_walker(rng3, 2, 300)
        merged = vcat(w1, w2)
        perm = sortperm(rand(stblrng(), length(eachindex(merged))))
        shuffled = merged[perm]

        # Independent series pool with their empirical mass fractions,
        # E_pool = 1 / Σ_j (α_j² / E_j):
        ess_direct = BAT._pooled_ess(
            [
                BAT._repetition_exact_ess(w1, EffSampleSizeFromAC(), context),
                BAT._repetition_exact_ess(w2, EffSampleSizeFromAC(), context)
            ],
            [sum(w1.weight), sum(w2.weight)]
        )
        ess_tagged = bat_eff_sample_size(shuffled, EffSampleSizeFromAC(), context).result
        @test ess_tagged ≈ ess_direct

        # Provenance-driven algorithm defaults: process ESS for tagged or
        # uniformly weighted samples, Kish ESS for nonuniformly weighted
        # samples without process provenance:
        @test bat_default(bat_eff_sample_size, Val(:algorithm), shuffled) isa EffSampleSizeFromAC
        untagged = DensitySampleVector(v = merged.v, logd = merged.logd, weight = float.(merged.weight))
        @test bat_default(bat_eff_sample_size, Val(:algorithm), untagged) isa KishESS
        uniformw = DensitySampleVector(v = merged.v, logd = merged.logd)
        @test bat_default(bat_eff_sample_size, Val(:algorithm), uniformw) isa EffSampleSizeFromAC
    end

    @testset "pooled ESS" begin
        # Mass-fraction pooling of independent series,
        # E_pool = 1 / Σ_j (α_j² / E_j): with equal masses, one series of
        # ESS 1000 and one of ESS 10 pool to about 39.6 - far below the
        # plain sum, which would hide the badly mixing series:
        @test BAT._pooled_ess([[1000.0], [10.0]], [1.0, 1.0]) ≈ [1 / (0.25 / 1000 + 0.25 / 10)]
        # Uniform efficiency reduces exactly to the sum:
        @test BAT._pooled_ess([[600.0, 60.0], [400.0, 40.0]], [600.0, 400.0]) ≈ [1000.0, 100.0]
        @test isnothing(BAT._pooled_ess(Any[], Float64[]))
    end

    @testset "weight-scale robustness" begin
        context = BATContext()
        mk_smpls(w) = DensitySampleVector(v = [rand(stblrng(), 2) for _ in eachindex(w)], logd = zeros(length(w)), weight = w)

        # Kish ESS must not overflow or lose scale invariance at extreme
        # weight scales:
        w = 1.0 .+ (1:100) ./ 100
        kish = bat_eff_sample_size(mk_smpls(w), KishESS(), context).result
        @test bat_eff_sample_size(mk_smpls(w .* 1e300), KishESS(), context).result ≈ kish
        @test bat_eff_sample_size(mk_smpls(w .* 1e-300), KishESS(), context).result ≈ kish
        @test bat_eff_sample_size(mk_smpls([fill(typemax(Int), 99); 4]), KishESS(), context).result ≈ 99 rtol = 0.01

        # The canonical relative weights reject invalid input:
        @test BAT._canonical_rel_weights([2, 4, 8]) ≈ [0.25, 0.5, 1.0]
        @test isempty(BAT._canonical_rel_weights(Float64[]))
        @test_throws ArgumentError BAT._canonical_rel_weights([0.0, 0.0])
        @test_throws ArgumentError BAT._canonical_rel_weights([1.0, -1.0])
        @test_throws ArgumentError BAT._canonical_rel_weights([1.0, Inf])
    end
end
