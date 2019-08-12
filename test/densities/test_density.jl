# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using ArraysOfArrays, Distributions, PDMats, StatsBase

struct test_density <: AbstractDensity
end

BAT.nparams(td::test_density) = Int(3)
BAT.sampler(td::test_density) = BAT.sampler(MvNormal(ones(3), PDMat(Matrix{Float64}(I,3,3))))

@testset "density" begin

    mvec = [-0.3, 0.3]
    cmat = [1.0 1.5; 1.5 4.0]
    Σ = @inferred PDMat(cmat)
    mvnorm = @inferred  MvNormal(mvec, Σ)

    density = let mvnorm = mvnorm
        @inferred GenericDensity(params -> logpdf(mvnorm, params), 2)
    end

    params = [0.0, 0.0]

    @testset "rand" begin
        td = test_density()
        @test rand(MersenneTwister(7002), td, Float64) ≈
            [-2.415270938, 0.7070171342, 1.0224848653]

        @test rand(MersenneTwister(7002), td, Float64, 2) ≈ VectorOfSimilarVectors([
            -2.415270938 1.2951273090;
            0.7070171342  0.8896838714;
            1.0224848653  0.8824274590;
        ])

        x = VectorOfSimilarVectors(ones(3, 2))
        @test x === @inferred rand!(MersenneTwister(7002), td, x)
        @test x ≈ VectorOfSimilarVectors([
            -2.415270938 1.2951273090;
            0.7070171342  0.8896838714;
            1.0224848653  0.8824274590;
        ])
    end

    @testset "param_bounds" begin
        pbounds = @inferred param_bounds(density)
        @test typeof(pbounds) == NoParamBounds
        @test pbounds.ndims == 2
    end

    @testset "density_logval" begin
        @test @inferred(density_logval(density, params)) ≈ logpdf(mvnorm, params)
        @test @inferred(density_logval(density, params)) ≈ logpdf(mvnorm, params)
    end

    @testset "parent" begin
        @test @inferred parent(density) == density.log_f
    end

    @testset "nparams" begin
        @test @inferred nparams(density) == 2
    end
end
