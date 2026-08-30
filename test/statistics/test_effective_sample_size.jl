# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Statistics
using ArraysOfArrays, ElasticArrays, StatsBase
using StableRNGs

function _ensemble_geyer_ess_oracle(values)
    n = length(values)
    centered = values .- mean(values)
    denominator = sum(abs2, centered)
    iszero(denominator) && return Float64(n)
    autocorrelation = [
        sum(@view(centered[1:(n - lag)]) .* @view(centered[(1 + lag):n])) /
            denominator
        for lag in 0:(n - 1)
    ]
    paired_sum = 0.0
    previous_pair = Inf
    i = 1
    while i < n - 1
        pair = min(autocorrelation[i] + autocorrelation[i + 1], previous_pair)
        pair >= 0 || break
        paired_sum += pair
        previous_pair = pair
        i += 2
    end
    return min(n, n / (-1 + 2 * paired_sum))
end

function _ensemble_ess_oracle(paths)
    nwalkers = length(paths)
    ensemble_mean = reduce(+, paths) ./ nwalkers
    return nwalkers .* [
        _ensemble_geyer_ess_oracle(@view ensemble_mean[d, :])
        for d in axes(ensemble_mean, 1)
    ]
end

function _pooled_ensemble_ess_oracle(ess_parts, masses)
    alpha = masses ./ sum(masses)
    return 1 ./ sum(alpha[i]^2 ./ ess_parts[i] for i in eachindex(ess_parts))
end

function _ensemble_walker_output(
    path, chainid, walkerid;
    compress = true, chaincycle = 1, step_offset = 0,
)
    run_starts = Int[1]
    weights = Int[1]
    for step in 2:size(path, 2)
        if compress && @view(path[:, step]) == @view(path[:, run_starts[end]])
            weights[end] += 1
        else
            push!(run_starts, step)
            push!(weights, 1)
        end
    end
    values = VectorOfSimilarVectors(path[:, run_starts])
    info = [
        BAT.MCMCSampleID(
            Int32(chainid), Int32(walkerid), Int32(chaincycle),
            Int64(step + step_offset), Int32(1), true,
        )
        for step in run_starts
    ]
    return DensitySampleVector(
        v = values,
        logd = zeros(eltype(path), length(run_starts)),
        weight = weights,
        info = info,
    )
end

function _ensemble_outputs(paths, chainid; compress = true)
    return [
        _ensemble_walker_output(path, chainid, walkerid; compress)
        for (walkerid, path) in pairs(paths)
    ]
end

@testset "effective_sample_size" begin
    stblrng() = StableRNG(789990641)

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

    @testset "coupled ensemble ESS" begin
        context = BATContext()
        signal = Float64[
            1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1,
            1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1,
        ]
        lagged = [0.0; signal[1:(end - 1)]]
        scalar_paths = [reshape(signal, 1, :), reshape(lagged, 1, :)]
        compressed_outputs = [_ensemble_outputs(scalar_paths, 1)]
        compressed_merged = reduce(vcat, only(compressed_outputs))
        expected = only(_ensemble_ess_oracle(scalar_paths))
        @test any(>(1), first(only(compressed_outputs)).weight)

        ensemble_ess = BAT._mcmc_ess(
            compressed_outputs, compressed_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        )
        @test ensemble_ess ≈ expected
        @test !isapprox(
            BAT._pooled_walker_ess(
                compressed_outputs, compressed_merged,
                RepetitionWeighting(), context,
            ),
            expected;
            rtol = 0.1,
        )

        unit_outputs = [_ensemble_outputs(scalar_paths, 1; compress = false)]
        unit_merged = reduce(vcat, only(unit_outputs))
        @test BAT._mcmc_ess(
            unit_outputs, unit_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        ) ≈ expected

        paths_1 = [
            vcat(reshape(signal, 1, :), reshape(cumsum(signal), 1, :)),
            vcat(
                reshape(lagged, 1, :),
                reshape([0.0; cumsum(signal)[1:(end - 1)]], 1, :),
            ),
        ]
        signal_2 = reverse(signal[1:20])
        lagged_2 = [0.0; signal_2[1:(end - 1)]]
        paths_2 = [
            vcat(
                reshape([0.0; cumsum(signal_2)[1:(end - 1)]], 1, :),
                reshape(lagged_2, 1, :),
            ),
            vcat(reshape(cumsum(signal_2), 1, :), reshape(signal_2, 1, :)),
        ]
        multiple_outputs = [
            _ensemble_outputs(paths_1, 1),
            _ensemble_outputs(paths_2, 2),
        ]
        multiple_merged = reduce(vcat, reduce(vcat, output) for output in multiple_outputs)
        ess_parts = [_ensemble_ess_oracle(paths_1), _ensemble_ess_oracle(paths_2)]
        masses = Float64[
            length(paths_1) * size(first(paths_1), 2),
            length(paths_2) * size(first(paths_2), 2),
        ]
        expected_pooled = _pooled_ensemble_ess_oracle(ess_parts, masses)
        premature_minimum = only(_pooled_ensemble_ess_oracle(
            [[minimum(part)] for part in ess_parts], masses,
        ))
        @test expected_pooled[1] != expected_pooled[2]
        @test minimum(expected_pooled) != premature_minimum
        @test BAT._mcmc_ess(
            multiple_outputs, multiple_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        ) ≈ minimum(expected_pooled)

        unequal_outputs = [[
            _ensemble_walker_output(scalar_paths[1], 1, 1),
            _ensemble_walker_output(scalar_paths[2][:, 1:(end - 1)], 1, 2),
        ]]
        unequal_merged = reduce(vcat, only(unequal_outputs))
        @test isnothing(BAT._mcmc_ess(
            unequal_outputs, unequal_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        ))

        shifted_outputs = [[
            _ensemble_walker_output(scalar_paths[1], 1, 1),
            _ensemble_walker_output(scalar_paths[2], 1, 2; step_offset = 1),
        ]]
        shifted_merged = reduce(vcat, only(shifted_outputs))
        @test isnothing(BAT._mcmc_ess(
            shifted_outputs, shifted_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        ))

        cross_cycle_outputs = [[
            _ensemble_walker_output(scalar_paths[1], 1, 1),
            _ensemble_walker_output(scalar_paths[2], 1, 2; chaincycle = 2),
        ]]
        wrong_chain_outputs = [[
            _ensemble_walker_output(scalar_paths[1], 1, 1),
            _ensemble_walker_output(scalar_paths[2], 2, 2),
        ]]
        duplicate_walker_outputs = [[
            _ensemble_walker_output(scalar_paths[1], 1, 1),
            _ensemble_walker_output(scalar_paths[2], 1, 1),
        ]]
        gapped_walker = _ensemble_walker_output(scalar_paths[2], 1, 2)
        gap_id = gapped_walker.info[2]
        gapped_walker.info[2] = BAT.MCMCSampleID(
            gap_id.chainid, gap_id.walkerid, gap_id.chaincycle,
            gap_id.stepno + 1, gap_id.proposalid, gap_id.sampletype,
        )
        gapped_outputs = [[
            _ensemble_walker_output(scalar_paths[1], 1, 1), gapped_walker,
        ]]
        for invalid_outputs in (
            cross_cycle_outputs, wrong_chain_outputs,
            duplicate_walker_outputs, gapped_outputs,
        )
            invalid_merged = reduce(vcat, only(invalid_outputs))
            @test isnothing(BAT._mcmc_ess(
                invalid_outputs, invalid_merged, StretchMove(),
                RepetitionWeighting(), false, context,
            ))
        end

        zero_run = _ensemble_walker_output(
            reshape([0.0], 1, :), 99, 99; chaincycle = 99, step_offset = 99,
        )
        zero_run.weight[1] = 0
        zero_weight_outputs = [[
            vcat(compressed_outputs[1][1], zero_run),
            vcat(compressed_outputs[1][2], zero_run),
        ]]
        zero_weight_merged = reduce(vcat, only(zero_weight_outputs))
        @test BAT._mcmc_ess(
            zero_weight_outputs, zero_weight_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        ) ≈ expected

        extreme_paths = [fill(2.0f38, 1, 16), fill(2.5f38, 1, 16)]
        extreme_outputs = [_ensemble_outputs(extreme_paths, 1)]
        extreme_merged = reduce(vcat, only(extreme_outputs))
        safe_mean = Float32((Float64(extreme_paths[1][1]) + Float64(extreme_paths[2][1])) / 2)
        mean_process = BAT._ensemble_mean_process(only(extreme_outputs), 16.0)
        @test eltype(flatview(mean_process)) == Float32
        @test all(==(safe_mean), flatview(mean_process))
        @test BAT._mcmc_ess(
            extreme_outputs, extreme_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        ) == 32

        for (T, nwalkers) in ((Float32, 10), (Float64, 11))
            limit = floatmax(T)
            for boundary in (limit, -limit)
                same_sign_outputs = _ensemble_outputs(
                    [fill(boundary, 1, 4) for _ in 1:nwalkers], 1,
                )
                same_sign_mean = BAT._ensemble_mean_process(same_sign_outputs, 4.0)
                @test eltype(flatview(same_sign_mean)) == T
                @test all(==(boundary), flatview(same_sign_mean))

                opposite_outputs = _ensemble_outputs(
                    [
                        fill(boundary, 1, 4), fill(-boundary, 1, 4),
                        fill(boundary, 1, 4),
                    ],
                    1,
                )
                opposite_mean = BAT._ensemble_mean_process(opposite_outputs, 4.0)
                opposite_expected = T(BigFloat(boundary) / 3)
                @test eltype(flatview(opposite_mean)) == T
                @test all(
                    x -> isapprox(x, opposite_expected; rtol = eps(T), atol = zero(T)),
                    flatview(opposite_mean),
                )
            end
        end
        @test isnothing(BAT._mcmc_ess(
            compressed_outputs, compressed_merged, StretchMove(),
            RepetitionWeighting(), true, context,
        ))
        @test isnothing(BAT._mcmc_ess(
            empty(compressed_outputs), compressed_merged[1:0], StretchMove(),
            RepetitionWeighting(), false, context,
        ))

        function constant_output(value, nsteps, walkerid)
            path = reshape(Float64[value, -value], 2, 1)
            output = _ensemble_walker_output(path, 1, walkerid)
            output.weight[1] = nsteps
            return output
        end
        one_buffer_outputs = [[constant_output(i, 70_000, i) for i in 1:4]]
        one_buffer_merged = reduce(vcat, only(one_buffer_outputs))
        @test BAT._mcmc_ess(
            one_buffer_outputs, one_buffer_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        ) == 280_000

        oversized_outputs = [[constant_output(i, 300_000, i) for i in 1:4]]
        oversized_merged = reduce(vcat, only(oversized_outputs))
        BAT._mcmc_ess(
            oversized_outputs, oversized_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        )
        @test isnothing(BAT._mcmc_ess(
            oversized_outputs, oversized_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        ))
        @test @allocated(BAT._mcmc_ess(
            oversized_outputs, oversized_merged, StretchMove(),
            RepetitionWeighting(), false, context,
        )) < 1_000_000

        old_ess = BAT._pooled_walker_ess(
            unit_outputs, unit_merged, RepetitionWeighting(), context,
        )
        @test BAT._mcmc_ess(
            unit_outputs, unit_merged, RandomWalk(),
            RepetitionWeighting(), false, context,
        ) ≈ old_ess
        @test BAT._mcmc_ess(
            unit_outputs, unit_merged, RandomWalk(),
            RepetitionWeighting(), true, context,
        ) ≈ old_ess
    end
end
