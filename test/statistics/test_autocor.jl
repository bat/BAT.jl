# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Statistics
using ArraysOfArrays, ElasticArrays, StatsBase
using StableRNGs

@testset "autocorr" begin
    stblrng() = StableRNG(789990641)

    # Primitive MCMC of flat distribution between [0, 0, 0] and [10, 5, 8]:
    rng = stblrng()
    v = nestedview(ElasticArray{Float64, 2}(undef, 3, 0))
    push!(v, [0, 0, 0])
    for i in 1:(10^4 - 1)
        push!(v, clamp.(last(v) .+ randn(rng, 3), [0, 0, 0], [10, 5, 8]))
    end

    v1 = flatview(v)[1, :]

    @testset "BAT._ac_next_pow_two" begin
        @test @inferred(BAT._ac_next_pow_two(0)) == 1
        @test @inferred(BAT._ac_next_pow_two(1)) == 1
        @test BAT._ac_next_pow_two(7) == 8
        @test BAT._ac_next_pow_two(8) == 8
    end

    @testset "BAT.fft_autocor" begin
        @test @inferred(BAT.fft_autocov(v)) isa ArrayOfSimilarArrays{Float64,1,1}
        @test flatview(BAT.fft_autocov(v)[1:20]) ≈ StatsBase.autocov(flatview(v)', 0:19)'

        @test @inferred(BAT.fft_autocov(v1)) isa Vector{Float64}
        @test BAT.fft_autocov(v1)[1:20] ≈ StatsBase.autocov(v1, 0:19)
    end

    @testset "BAT.fft_autocor" begin
        @test @inferred(BAT.fft_autocor(v)) isa ArrayOfSimilarArrays{Float64,1,1}
        @test flatview(BAT.fft_autocor(v)[1:20]) ≈ StatsBase.autocor(flatview(v)', 0:19)'

        @test @inferred(BAT.fft_autocor(v1)) isa Vector{Float64}
        @test BAT.fft_autocor(v1)[1:20] ≈ StatsBase.autocor(v1, 0:19)
    end
end
