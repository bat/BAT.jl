# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, MeasureBase
using Distributions, StatsBase, IntervalSets

@testset "truncate_batmeasure" begin
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
    xs = rand(m^100)
    xs_logd = logdensityof(m).(xs)
    smpls = DensitySampleVector(v = xs, logd = xs_logd)
    empirical_m = DensitySampleMeasure(smpls, getdof(m))

    em2 = EvaluatedMeasure(m, empirical = empirical_m, mass = 1)
end
