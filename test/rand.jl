# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test

using Distributions


@testset "random number generation" begin
    @testset "rand_gamma" begin
        import BAT.rand_gamma
        @test rand_gamma(MersenneTwister(7002), Float64, 0.5, 20.3) ≈ 0.872195443488103
        @test rand_gamma(MersenneTwister(8456), Float64, 0.5, 20.3) ≈ 0.2723580272080763
        @test rand_gamma(MersenneTwister(3296), Float64, 0.5, 20.3) ≈ 6.984079300633383
        @test rand_gamma(MersenneTwister(5065), Float64, 0.5, 20.3) ≈ 9.387736826614171
        @test rand_gamma(MersenneTwister(7002), Float64, 0.25, 2) ≈ 0.0031794482969992074
        @test rand_gamma(MersenneTwister(7002), Float64, 10, 2) ≈ 4.923284832865044
        @test typeof(rand_gamma(MersenneTwister(7002), Float32, 4.2)) == Float32
        @test typeof(rand_gamma(MersenneTwister(7002), Float32, 4.2, 2.2)) == Float32
    end

    @testset "rand_chisq" begin
        import BAT.rand_chisq
        @test rand_chisq(MersenneTwister(7002), Float64, 0.5) ≈ 0.0031794482969992074
        @test rand_chisq(MersenneTwister(7002), Float64, 20) ≈ 4.923284832865044
        @test typeof(rand_chisq(MersenneTwister(7002), Float32, 20)) == Float32
    end

    @testset "DistForRNG" begin
        @testset "DistForRNG{Normal}" begin
            @test rand(DistForRNG(Normal(1.7, 3.2)), MersenneTwister(2798)) == 8.504866599763968
            @test typeof(rand(DistForRNG(Normal(1.7, 3.2)), Float32)) == Float32
            @test issymmetric_around_origin(DistForRNG(Normal(0.0, 3.2))) == true
            @test issymmetric_around_origin(DistForRNG(Normal(1.7, 3.2))) == false
        end

        @testset "DistForRNG{Gamma}" begin
            @test rand(DistForRNG(Gamma(0.5, 20.3)), MersenneTwister(7002)) ≈ 0.872195443488103
            @test typeof(rand(DistForRNG(Gamma(4.2, 2.2)), Float32)) == Float32
            @test issymmetric_around_origin(DistForRNG(Gamma(4.2, 2.2))) == false
        end

        @testset "DistForRNG{Chisq}" begin
            @test rand(DistForRNG(Chisq(0.5)), MersenneTwister(7002)) ≈ 0.0031794482969992074
            @test typeof(rand(DistForRNG(Chisq(20.3)), Float32)) == Float32
            @test issymmetric_around_origin(DistForRNG(Chisq(20.3))) == false
        end
    end
end
