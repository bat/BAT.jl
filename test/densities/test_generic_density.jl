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
        @inferred GenericDensity(params -> logpdf(mvnorm, params), 2)
    end

    params = [0.0, 0.0]

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
