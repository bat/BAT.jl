# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using Distributions, PDMats, StatsBase


@testset "gamma_dist" begin
    @testset "BATGammaMTSampler" begin
        @test typeof(@inferred bat_sampler(Gamma(0.5, 20.3))) <: BATGammaMTSampler

        @test rand(MersenneTwister(7002), BATGammaMTSampler(Gamma(0.5, 20.3))) ≈ 0.872195443488103
        @test rand(MersenneTwister(8456), BATGammaMTSampler(Gamma(0.5, 20.3))) ≈ 0.2723580272080763
        @test rand(MersenneTwister(3296), BATGammaMTSampler(Gamma(0.5, 20.3))) ≈ 6.984079300633383
        @test rand(MersenneTwister(5065), BATGammaMTSampler(Gamma(0.5, 20.3))) ≈ 9.387736826614171
        @test rand(MersenneTwister(7002), BATGammaMTSampler(Gamma(0.25, 2))) ≈ 0.0031794482969992074
        @test rand(MersenneTwister(7002), BATGammaMTSampler(Gamma(10, 2))) ≈ 4.923284832865044

        @test typeof(rand(BATGammaMTSampler(Gamma(4.2f0, 2.2f0)))) == Float32
        @test eltype(BATGammaMTSampler(Gamma(4.2f0, 2.2f0))) == Float32
        @test eltype(BATGammaMTSampler(Gamma(4.2, 2.2))) == Float64
        @test typeof(rand(BATGammaMTSampler(Gamma(4.2, 2.2)))) == Float64
    end

    @testset "rand" begin
        bsguv = BATGammaMTSampler(Gamma(4f0, 2f0))
        @test size(rand(bsguv, 5)) == (5,)
        @test typeof(rand(bsguv, 5)) == Vector{Float32}
        @test typeof(rand(bsguv)) == Float32

        res = rand!(bsguv, zeros(3))
        @test size(res) == (3,)
        @test typeof(res) == Array{Float64, 1}
        res = rand!(bsguv, zeros(3,4))
        @test size(res) == (3,4,)
        @test typeof(res) == Array{Float64, 2}

        res = rand(Random.GLOBAL_RNG, bsguv, Dims((2,3)))
        @test typeof(res) == Array{Float32,2}
        @test size(res) == (2,3)
    end
end
