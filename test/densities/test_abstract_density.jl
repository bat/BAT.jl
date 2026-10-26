# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, StableRNGs
using DensityInterface, ValueShapes
using ArraysOfArrays, Distributions, PDMats, StatsBase
using AutoDiffOperators
import Zygote


@testset "abstract_density" begin
    @testset "Zygote zero tangent" begin
        adsel = BAT.get_adselector(BATContext(ad = Zygote))

        uniform = unshaped(batmeasure(Uniform(-1, 1)))
        uniform_valgrad = valgrad_func(BAT.checked_logdensityof(uniform), adsel, [0.2])
        @test uniform_valgrad([0.2]) == (-log(2), [0.0])
    end

    @testset "checked_logdensityof" begin
        d_nan = logfuncdensity(x -> NaN)
        d_inf = logfuncdensity(x -> +Inf)

        # A NaN density at a non-finite variate means no probability mass
        # there, everything else is a model error:
        @test BAT.checked_logdensityof(d_nan, [Inf, 1.0]) == -Inf
        @test_throws BAT.EvalException BAT.checked_logdensityof(d_nan, [0.5, 1.0])
        @test_throws BAT.EvalException BAT.checked_logdensityof(d_inf, [Inf, 1.0])
        @test_throws BAT.EvalException BAT.checked_logdensityof(d_inf, [0.5, 1.0])
    end

    @testset "checked_logdensities" begin
        d_nan = logfuncdensity(x -> NaN)
        d_inf = logfuncdensity(x -> +Inf)

        # Same policy as checked_logdensityof, applied per variate:
        @test BAT.checked_logdensities(d_nan, [[Inf, 1.0], [-Inf, 2.0]]) == [-Inf, -Inf]
        @test_throws BAT.EvalException BAT.checked_logdensities(d_nan, [[Inf, 1.0], [0.5, 1.0]])
        @test_throws BAT.EvalException BAT.checked_logdensities(d_inf, [[Inf, 1.0]])

        m = unshaped(batmeasure(MvNormal([1.0, 2.0], [2.0 1.5; 1.5 3.0])))
        X = nestedview(randn(StableRNG(7), 2, 5))
        @test BAT.checked_logdensities(m, X) ≈ BAT.checked_logdensityof.(m, X)
        @test BAT.checked_logdensities(m, collect(X)) ≈ BAT.checked_logdensityof.(m, X)
    end
end
