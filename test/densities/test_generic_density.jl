# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using ArraysOfArrays, Distributions, PDMats, StatsBase


@testset "generic_density" begin
    mvec = [-0.3, 0.3]
    cmat = [1.0 1.5; 1.5 4.0]
    Σ = @inferred PDMat(cmat)
    mvnorm = @inferred  MvNormal(mvec, Σ)

    density = let mvnorm = mvnorm
        @inferred BAT.GenericDensity(params -> LogDVal(logpdf(mvnorm, params)))
    end

    params = [0.0, 0.0]

    @testset "BAT.eval_logval_unchecked" begin
        @test @inferred(BAT.eval_logval_unchecked(density, params)) ≈ logpdf(mvnorm, params)
        @test @inferred(BAT.eval_logval_unchecked(density, params)) ≈ logpdf(mvnorm, params)
    end

    @testset "parent" begin
        @test @inferred parent(density) == density.f
    end
end
