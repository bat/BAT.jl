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

    @testset "bat_autocorr" begin
        @test @inferred(bat_autocorr(v)).result isa ArrayOfSimilarArrays{Float64,1,1}
        acf = bat_autocorr(v).result
        @test length(acf) == length(v)

  
        @test @inferred(bat_autocorr(v1)).result isa Vector{Float64}
        @test flatview(acf)[1,:] ≈ bat_autocorr(v1).result
  
  
        @test flatview(acf)[1, [1, 10, 20, 30]] ≈ [1.0, 0.7020507052044521, 0.47068454095108886, 0.3294865638787076]
        @test sum(acf) ≈ fill(0.5, 3)

        taus = 2 * cumsum(flatview(acf)[1, :]) .- 1
        @test @inferred(BAT.emcee_auto_window(taus, 5)) == 222
    end

    @testset "BAT.emcee_integrated_time" begin
        @inferred(bat_integrated_autocorr_len(v1)).result ≈ 44.28039926162039
        @test_throws ErrorException bat_integrated_autocorr_len(v1[1:100])
        @inferred(bat_integrated_autocorr_len(v)).result ≈ [44.28039926162039, 16.83675738165955, 32.05622635001912]
        @test_throws ErrorException bat_integrated_autocorr_len(v[1:100])
    end
end
