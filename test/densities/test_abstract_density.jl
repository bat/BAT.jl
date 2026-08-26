# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, StableRNGs
using DensityInterface, ValueShapes
using ArraysOfArrays, Distributions, PDMats, StatsBase
using AutoDiffOperators
import Zygote


@testset "abstract_density" begin
    @testset "Zygote zero tangents" begin
        adsel = BAT.get_adselector(BATContext(ad = Zygote))

        uniform = unshaped(batmeasure(Uniform(-1, 1)))
        uniform_valgrad = valgrad_func(BAT.checked_logdensityof(uniform), adsel, [0.2])
        @test uniform_valgrad([0.2]) == (-log(2), [0.0])
        @test uniform_valgrad([1.5]) == (-Inf, [0.0])

        truncated_normal = unshaped(batmeasure(truncated(Normal(), -1, 1)))
        truncated_valgrad = valgrad_func(BAT.checked_logdensityof(truncated_normal), adsel, [0.2])
        @test last(truncated_valgrad([0.2])) ≈ [-0.2]
        @test truncated_valgrad([1.5]) == (-Inf, [0.0])
    end
end
