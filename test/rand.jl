# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test


@testset "random number generation" begin
    @testset "rand_gamma" begin
        import BAT.rand_gamma
        @test rand_gamma(MersenneTwister(7002), Float64, 0.5, 20.3) ≈ 0.872195443488103
        @test rand_gamma(MersenneTwister(8456), Float64, 0.5, 20.3) ≈ 0.2723580272080763
        @test rand_gamma(MersenneTwister(3296), Float64, 0.5, 20.3) ≈ 6.984079300633383
        @test rand_gamma(MersenneTwister(5065), Float64, 0.5, 20.3) ≈ 9.387736826614171
        @test rand_gamma(MersenneTwister(7002), Float64, 0.25, 2) ≈ 0.0031794482969992074
        @test rand_gamma(MersenneTwister(7002), Float64, 10, 2) ≈ 4.923284832865044
    end

    @testset "rand_chisq" begin
        import BAT.rand_chisq
        @test rand_chisq(MersenneTwister(7002), Float64, 0.5) ≈ 0.0031794482969992074
        @test rand_chisq(MersenneTwister(7002), Float64, 20) ≈ 4.923284832865044
    end
end
