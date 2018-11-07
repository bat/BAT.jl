# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using Distributions, PDMats, StatsBase


@testset "chisq_dist" begin
    @testset "BATChisqSampler" begin
        @test typeof(@inferred bat_sampler(Chisq(0.5))) <: BATChisqSampler

        @test rand(MersenneTwister(7002), BATChisqSampler(Chisq(0.5))) ≈ 0.0031794482969992074
        @test rand(MersenneTwister(7002), BATChisqSampler(Chisq(20))) ≈ 4.923284832865044

        @test eltype(BATChisqSampler(Chisq(20f0))) == Float32
        @test typeof(rand(BATChisqSampler(Chisq(20f0)))) == Float32
    end
end
