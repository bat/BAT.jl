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
        @test @inferred(bat_integrated_autocorr_len(v1, GeyerAutocorLen())).result ≈ 52.2404651916953
        @test @inferred(bat_integrated_autocorr_len(v, GeyerAutocorLen())).result ≈ [52.240465191695314, 17.04353447359818, 38.393838710754345]

        @test @inferred(bat_integrated_autocorr_len(v1, SokalAutocorLen())).result ≈ 44.243392655975356
        @test @inferred(bat_integrated_autocorr_len(v, SokalAutocorLen())).result ≈ [44.243392655975356, 16.794891919657566, 31.94870020972804]
    end
end
