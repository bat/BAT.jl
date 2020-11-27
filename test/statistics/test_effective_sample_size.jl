# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Distributions
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

    bsample = @inferred(bat_sample(product_distribution([Normal() for i in 1:3]), 10^4)).result
    @test isapprox(bat_integrate(bsample).result.val, 1.0, rtol=15)
    eff_sample_size_dsample_vec = @inferred(bat_eff_sample_size(bsample)).result
    eff_sample_size_arr_of_sim_arr = @inferred(bat_eff_sample_size(bsample.v)).result

    [bsample.weight[i] = 0 for i in eachindex(bsample.weight)]
    bsample.weight[1] = 1
    eff_sample_size_dsample_vec_single_weight = @inferred(bat_eff_sample_size(bsample)).result

    for i in eachindex(@inferred(bat_eff_sample_size(bsample)).result)
        @test 8000 <= eff_sample_size_dsample_vec[i] <= 10^4
        @test eff_sample_size_dsample_vec[i] == eff_sample_size_arr_of_sim_arr[i]
    end

    for i in eachindex(eff_sample_size_dsample_vec_single_weight)
        @test @inferred(isapprox(1, eff_sample_size_dsample_vec_single_weight[i]))
    end

    @testset "BAT.bat_integrated_autocorr_len" begin
        @test @inferred(bat_integrated_autocorr_len(v1, GeyerAutocorLen())).result ≈ 52.2404651916953
        @test @inferred(bat_integrated_autocorr_len(v, GeyerAutocorLen())).result ≈ [52.240465191695314, 17.04353447359818, 38.393838710754345]

        @test @inferred(bat_integrated_autocorr_len(v1, SokalAutocorLen())).result ≈ 44.243392655975356
        @test @inferred(bat_integrated_autocorr_len(v, SokalAutocorLen())).result ≈ [44.243392655975356, 16.794891919657566, 31.94870020972804]
    end
end
