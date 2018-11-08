# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using Distributions, PDMats, StatsBase


@testset "t_dist" begin
    @testset "BATMvTDistSampler" begin
        d = MvTDist(1.5, PDMat([2.0 1.0; 1.0 3.0]))

        @test typeof(@inferred bat_sampler(d)) <: BATMvTDistSampler
        @test size(@inferred rand(bat_sampler(d), 5)) == (2, 5)

        @test size(rand!(bat_sampler(d), ones(2))) == (2, )
        @test size(rand!(bat_sampler(d), ones(2, 3))) == (2, 3)

        cmat = [3.76748 0.446731 0.625418; 0.446731 3.9317 0.237361; 0.625418 0.237361 3.43867]
        tmean = [1., 2, 3]
        tmv = MvTDist(3, tmean, PDMat(Matrix{Float64}(I, 3, 3)))

        tmv2 = BAT.set_cov(tmv, cmat)
        @test Matrix(BAT.get_cov(tmv2)) ≈ cmat

        bstmv = BATMvTDistSampler(tmv2)

        n = 1000
        res = rand(MersenneTwister(7002), bstmv, n)

        @test isapprox(mean(res,dims=2), tmean; atol = 0.5)
        @test isapprox(cov(res, dims=2)/3, cmat; atol = 1.5)
    end

    @testset "rand" begin
        bstmv = BAT.BATMvTDistSampler(MvTDist(1.5, PDMat(Matrix{Float64}(I, 3, 3))))
        res = rand(bstmv)
        @test typeof(res) == Array{Float64, 1}
        @test size(res) == (3,)
        res = rand(bstmv, 2)
        @test typeof(res) == Array{Float64, 2}
        @test size(res) == (3, 2)
    end
end
