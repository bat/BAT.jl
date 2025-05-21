# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Statistics, LinearAlgebra
using Distributions, PDMats
using StableRNGs
using InverseFunctions


@testset "polar_shell_distribution" begin
    # ToDo: Improve test coverage, test shape of generated samples

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
