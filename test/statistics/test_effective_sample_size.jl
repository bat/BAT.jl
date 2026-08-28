# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Statistics
using ArraysOfArrays, ElasticArrays, StatsBase
using LogarithmicNumbers: ULogarithmic
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
        vals = VectorOfSimilarVectors(ElasticArray{Float64, 2}(undef, 2, 0))
        push!(vals, [0.0, 0.0])
        for _ in 1:(n_runs - 1)
            push!(vals, last(vals) .+ randn(rng2, 2))
        end
        weights = rand(rng2, 1:5, n_runs)
        smpls_rle = DensitySampleVector(v = vals, logd = zeros(n_runs), weight = weights)

        expanded = VectorOfSimilarVectors(flatview(vals)[:, inverse_rle(1:n_runs, weights)])
        ess_rle = BAT._repetition_exact_ess(smpls_rle, EffSampleSizeFromAC(), context)
        ess_expanded = bat_eff_sample_size(expanded, EffSampleSizeFromAC(), context).result
        @test ess_rle ≈ ess_expanded

        # Uniform repetition counts greater than one also decode:
        smpls_unif = DensitySampleVector(v = vals, logd = zeros(n_runs), weight = fill(2, n_runs))
        expanded_unif = VectorOfSimilarVectors(flatview(vals)[:, inverse_rle(1:n_runs, fill(2, n_runs))])
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
            vals_w = VectorOfSimilarVectors(ElasticArray{Float64, 2}(undef, 2, 0))
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

        # Provenance takes priority over the uniform-weight fast path:
        # merged unit-weight chains must not be treated as one series
        # across chain boundaries, so the result is permutation-invariant
        # and matches the pooled per-chain ESS:
        u1 = DensitySampleVector(v = w1.v, logd = w1.logd, info = w1.info)
        u2 = DensitySampleVector(v = w2.v, logd = w2.logd, info = w2.info)
        umerged = vcat(u1, u2)
        ushuffled = umerged[perm]
        ess_upooled = BAT._pooled_ess(
            [
                bat_eff_sample_size(u1.v, EffSampleSizeFromAC(), context).result,
                bat_eff_sample_size(u2.v, EffSampleSizeFromAC(), context).result
            ],
            [sum(u1.weight), sum(u2.weight)]
        )
        @test bat_eff_sample_size(umerged, EffSampleSizeFromAC(), context).result ≈ ess_upooled
        @test bat_eff_sample_size(ushuffled, EffSampleSizeFromAC(), context).result ≈ ess_upooled
        for scale in (floatmax(Float64), big"1e10000")
            scaled = DensitySampleVector(
                v = umerged.v,
                logd = umerged.logd,
                weight = fill(scale, length(umerged)),
                info = umerged.info,
            )
            @test bat_eff_sample_size(scaled, EffSampleSizeFromAC(), context).result ≈ ess_upooled
        end

        walker_outputs = [u1, u2]
        walker_ess = BAT._pooled_walker_ess([walker_outputs], umerged, ARPWeighting(), context)
        for scale in (floatmax(Float64), big"1e10000")
            scaled_outputs = [[
                DensitySampleVector(
                    v = output.v,
                    logd = output.logd,
                    weight = fill(scale, length(output)),
                    info = output.info,
                )
                for output in walker_outputs
            ]]
            scaled_merged = reduce(vcat, only(scaled_outputs))
            @test BAT._pooled_walker_ess(scaled_outputs, scaled_merged, ARPWeighting(), context) ≈ walker_ess
        end

        # A uniform repetition weight > 1 on a tagged chain decodes to
        # the expanded ordered chain:
        n_u = length(eachindex(u1))
        t2 = DensitySampleVector(v = u1.v, logd = u1.logd, weight = fill(2, n_u), info = u1.info)
        expanded_t2 = VectorOfSimilarVectors(flatview(u1.v)[:, inverse_rle(1:n_u, fill(2, n_u))])
        @test bat_eff_sample_size(t2, EffSampleSizeFromAC(), context).result ≈
            bat_eff_sample_size(expanded_t2, EffSampleSizeFromAC(), context).result

        tagged_inf = DensitySampleVector(v = u1.v, logd = u1.logd, weight = fill(Inf, n_u), info = u1.info)
        @test_throws ArgumentError bat_eff_sample_size(tagged_inf, EffSampleSizeFromAC(), context)

        # Exact repeats created by systematic resampling retain process
        # provenance, independent of the input storage order:
        function repeated_walker(chainid, offset)
            n = 200
            vals = [[sin(i / 8) + offset] for i in 1:n]
            ids = [BAT.MCMCSampleID(Int32(chainid), Int32(1), Int32(1), Int64(i), Int32(1), true) for i in 1:n]
            DensitySampleVector(v = vals, logd = zeros(n), weight = fill(2, n), info = ids)
        end
        repeated = vcat(repeated_walker(1, 0.0), repeated_walker(2, 0.15))
        repeated_perm = sortperm(rand(StableRNG(33), length(repeated)))
        resampling = SystematicResampling(nsamples = 800)
        resampled_ordered = samplesof(evalmeasure(repeated, resampling, BATContext(rng = StableRNG(44))))
        resampled_shuffled = samplesof(evalmeasure(repeated[repeated_perm], resampling, BATContext(rng = StableRNG(44))))
        for resampled in (resampled_ordered, resampled_shuffled)
            @test length(resampled) == 800
            @test all(isone, resampled.weight)
            @test BAT._has_process_provenance(resampled)
            @test only(bat_eff_sample_size(resampled, EffSampleSizeFromAC(), context).result) ≈ 24.588625082911854
        end

        # Provenance-driven algorithm defaults: process ESS for tagged or
        # uniformly weighted samples, Kish ESS for nonuniformly weighted
        # samples without process provenance:
        @test bat_default(bat_eff_sample_size, Val(:algorithm), shuffled) isa EffSampleSizeFromAC
        untagged = DensitySampleVector(v = merged.v, logd = merged.logd, weight = float.(merged.weight))
        @test bat_default(bat_eff_sample_size, Val(:algorithm), untagged) isa KishESS
        uniformw = DensitySampleVector(v = merged.v, logd = merged.logd)
        @test bat_default(bat_eff_sample_size, Val(:algorithm), uniformw) isa EffSampleSizeFromAC
        # Degenerate (non-unique) sample ids are unusable provenance, so
        # nonuniform weights fall back to Kish:
        dup_ids = fill(BAT.MCMCSampleID(Int32(1), Int32(1), Int32(1), Int64(1), Int32(1), true), length(eachindex(merged)))
        degenerate = DensitySampleVector(v = merged.v, logd = merged.logd, weight = float.(merged.weight), info = dup_ids)
        @test !BAT._has_process_provenance(degenerate)
        @test bat_default(bat_eff_sample_size, Val(:algorithm), degenerate) isa KishESS
    end

    @testset "pooled ESS" begin
        # Mass-fraction pooling of independent series,
        # E_pool = 1 / Σ_j (α_j² / E_j): with equal masses, one series of
        # ESS 1000 and one of ESS 10 pool to about 39.6 - far below the
        # plain sum, which would hide the badly mixing series:
        @test BAT._pooled_ess([[1000.0], [10.0]], [1.0, 1.0]) ≈ [1 / (0.25 / 1000 + 0.25 / 10)]
        # Uniform efficiency reduces exactly to the sum:
        @test BAT._pooled_ess([[600.0, 60.0], [400.0, 40.0]], [600.0, 400.0]) ≈ [1000.0, 100.0]

        ess_parts = [[3.0], [5.0]]
        k = typemax(Int) ÷ 5
        for masses in (
            [3.0, 5.0],
            (floatmax(Float64) / 5) .* [3, 5],
            k .* [3, 5],
        )
            @test BAT._pooled_ess(ess_parts, masses) ≈ [8.0]
        end
        @test BAT._pooled_ess(reverse(ess_parts), [5.0, 3.0]) ≈ [8.0]
        @test BAT._pooled_ess([[0.0], [5.0]], [0.0, 5.0]) ≈ [5.0]
        @test BAT._pooled_ess([[0.0], [5.0]], [1.0, 1.0]) == [0.0]
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
        @test bat_eff_sample_size(mk_smpls(fill(Float16(1), 300)), KishESS(), context).result ≈ 300

        ac = bat_eff_sample_size(mk_smpls(ones(Int, 100)), EffSampleSizeFromAC(), context).result
        @test bat_eff_sample_size(mk_smpls(fill(typemax(Int), 100)), EffSampleSizeFromAC(), context).result ≈ ac
        @test_throws ArgumentError bat_eff_sample_size(mk_smpls(fill(Inf, 100)), EffSampleSizeFromAC(), context)

        # The canonical relative weights reject invalid input:
        @test BAT._canonical_rel_weights([2, 4, 8]) ≈ [0.25, 0.5, 1.0]
        @test BAT._canonical_rel_weights(Float16[1, 3]) == Float32[1 / 3, 1]
        @test first(BAT._canonical_rel_weights(Real[nextfloat(Float16(0)), floatmax(Float16)])) > 0
        @test first(BAT._canonical_rel_weights(exp.(ULogarithmic, Float16[-20, 0]))) == exp(Float32(-20))
        @test eltype(BAT._canonical_rel_weights(BigFloat[1, 2])) === BigFloat
        @test isempty(BAT._canonical_rel_weights(Float64[]))
        @test_throws ArgumentError BAT._canonical_rel_weights([0.0, 0.0])
        @test_throws ArgumentError BAT._canonical_rel_weights([1.0, -1.0])
        @test_throws ArgumentError BAT._canonical_rel_weights([1.0, Inf])
    end
end
