# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using ArraysOfArrays, Distributions, PDMats, StatsBase


@testset "abstract_density" begin
    struct test_density <: AbstractDensity
    end

    BAT.nparams(td::test_density) = Int(3)
    BAT.sampler(td::test_density) = BAT.sampler(MvNormal(ones(3), PDMat(Matrix{Float64}(I,3,3))))

    @testset "rand" begin
        td = test_density()
        @test rand(MersenneTwister(7002), sampler(td)) ≈ [-2.415270938, 0.7070171342, 1.0224848653]
    end

    # ToDo: Add more tests
end
