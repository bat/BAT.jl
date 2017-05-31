# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test

using Distributions, PDMats


@testset "random number generation" begin
    @testset "rand" begin
        @test size(rand(bat_sampler(Gamma(4f0, 2f0)), 5)) == (5,)
        @test typeof(rand(bat_sampler(Gamma(4f0, 2f0)), 5)) == Vector{Float32}
    end


    @testset "issymmetric_around_origin" begin
        @test issymmetric_around_origin(Normal(0.0, 3.2)) == true
        @test issymmetric_around_origin(Normal(1.7, 3.2)) == false
        @test issymmetric_around_origin(Gamma(4.2, 2.2)) == false
        @test issymmetric_around_origin(Chisq(20.3)) == false
    end


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


    @testset "BATChisqSampler" begin
        @test typeof(@inferred bat_sampler(Chisq(0.5))) <: BATChisqSampler

        @test rand(MersenneTwister(7002), BATChisqSampler(Chisq(0.5))) ≈ 0.0031794482969992074
        @test rand(MersenneTwister(7002), BATChisqSampler(Chisq(20))) ≈ 4.923284832865044

        @test eltype(BATChisqSampler(Chisq(20f0))) == Float32
        @test typeof(rand(BATChisqSampler(Chisq(20f0)))) == Float32
    end


    @testset "BATMvTDistSampler" begin
        d = MvTDist(1.5, PDMat([2.0 1.0; 1.0 3.0]))

        @test typeof(@inferred bat_sampler(d)) <: BATMvTDistSampler
        @test size(@inferred rand(bat_sampler(d), 5)) == (2, 5)
    end
end
