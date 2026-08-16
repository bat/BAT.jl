# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Statistics, LinearAlgebra
using Distributions, PDMats
using StableRNGs
using InverseFunctions


@testset "polar_shell_distribution" begin
    # ToDo: Improve test coverage, test shape of generated samples

    @testset "constructor validates base dimension" begin
        @test_throws ArgumentError BAT.PolarShellDistribution(MvNormal(zeros(1), Diagonal(ones(1))))

        d_2d = BAT.PolarShellDistribution(MvNormal(zeros(2), Diagonal(ones(2))))
        @test length(rand(StableRNG(42), d_2d)) == 2

        @test_throws ArgumentError BAT.PolarShellDistribution(MvNormal(zeros(3), Diagonal(ones(3))))
    end

    base_dist = MvNormal([1,1], Diagonal([1,1]))
    base_dist = MvNormal(Diagonal([1,1]))
    d = BAT.PolarShellDistribution(base_dist)

    @test rand(d, 10^5) isa AbstractMatrix

    x = rand(d)
    
    @test @inferred(logpdf(d, x)) isa Real
    @test log(@inferred(pdf(d, x))) ≈ logpdf(d, x)

    f_tr = BAT.DistributionTransform(Normal, d)
    y = @inferred(f_tr(x))
    @test @inferred(inverse(f_tr)(y)) ≈ x


    f_tr = BAT.DistributionTransform(Uniform, d)
    y = @inferred(f_tr(x))
    @test @inferred(inverse(f_tr)(y)) ≈ x
end
