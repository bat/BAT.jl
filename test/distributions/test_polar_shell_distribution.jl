# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface

using Random, Statistics, LinearAlgebra
using Distributions, PDMats
using StableRNGs
using InverseFunctions
using MeasureBase: StdUniform, StdNormal, transport_to


@testset "polar_shell_distribution" begin
    # ToDo: Improve test coverage, test shape of generated samples

    base_dist = MvNormal([1,1], Diagonal([1,1]))
    base_dist = MvNormal(Diagonal([1,1]))
    d = BAT.PolarShellDistribution(base_dist)

    @test rand(d, 10^5) isa AbstractMatrix

    x = rand(d)
    
    @test @inferred(logpdf(d, x)) isa Real
    @test log(@inferred(pdf(d, x))) ≈ logpdf(d, x)

    m = batmeasure(d)
    @test logdensityof(m, x) ≈ logpdf(d, x)

    for ν in (StdNormal()^2, StdUniform()^2)
        f_tr = transport_to(ν, m)
        y = @inferred(f_tr(x))
        @test @inferred(inverse(f_tr)(y)) ≈ x
    end
end
