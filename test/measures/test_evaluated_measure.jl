# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using DensityInterface, MeasureBase, ValueShapes
using Distributions

@testset "evaluated_measure" begin
    dist = distprod(a = truncated(Normal(), -2, 2), b = Exponential())

    m = batmeasure(dist)

    @test BAT.unevaluated(EvaluatedMeasure(m)) === m

    n = 100
    xs = rand(Xoshiro(564003), m^n)
    xs_logd = logdensityof(m).(xs)
    smpls = DensitySampleVector(v = xs, logd = xs_logd)
    empirical_m = DensitySampleMeasure(smpls, dof = getdof(m))

    em = EvaluatedMeasure(m, empirical = empirical_m, mass = 1)
    @test BAT.unevaluated(em) === m
    @test BAT.empiricalof(em) === empirical_m
    @test BAT.samplesof(em) == smpls
    @test massof(em) ≈ 1
    x = first(xs)
    @test logdensityof(em, x) == logdensityof(m, x)
    @test DensitySampleVector(em) == smpls

    em_plain = EvaluatedMeasure(m)
    @test BAT.empiricalof(em_plain) === nothing
end
