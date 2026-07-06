# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using DensityInterface, MeasureBase, ValueShapes
using Distributions, StatsBase, IntervalSets

@testset "evaluated_measure" begin
    dist = distprod(
        a = truncated(Normal(), -2, 2),
        b = Exponential(),
        c = [1 2; 3 4],
        d = [-3..3, -4..4]
    )

    m = batmeasure(dist)

    @test @inferred(BAT.unevaluated(EvaluatedMeasure(m))) === m
    @test @inferred(BAT.unevaluated(EvaluatedMeasure(dist))).dist === dist

    n = 100
    xs = rand(Random.default_rng(), m^n)
    xs_logd = logdensityof(m).(xs)
    smpls = DensitySampleVector(v = xs, logd = xs_logd)
    empirical_m = DensitySampleMeasure(smpls, dof = getdof(m))

    em = EvaluatedMeasure(m, empirical = empirical_m, mass = 1)
    @test @inferred(BAT.unevaluated(em)) === m
    @test @inferred(BAT.empiricalof(em)) === empirical_m
    @test @inferred(BAT.samplesof(em)) === BAT.samplesof(empirical_m)
    @test @inferred(getdof(em)) == getdof(m)
    @test massof(em) ≈ 1
    @test @inferred(varshape(em)) == varshape(m)
    x = first(xs)
    @test @inferred(logdensityof(em, x)) == logdensityof(m, x)
    @test DensitySampleVector(em) == smpls

    em_dsm = EvaluatedMeasure(batmeasure(smpls))
    @test BAT.empiricalof(em_dsm) === BAT.unevaluated(em_dsm)
end
