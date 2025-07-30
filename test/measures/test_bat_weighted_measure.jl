# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, Distributions, MeasureBase, ValueShapes
using Statistics, StatsBase, IntervalSets
using MeasureBase: weightedmeasure

@testset "bat_weighted_measure" begin
    parent_dist = NamedTupleDist(a = Normal(), b = Weibull())
    vs = varshape(parent_dist)
    logweight = 4.2
    parent_measure = batmeasure(parent_dist)

    @test @inferred(weightedmeasure(logweight, parent_measure)) isa BAT.BATWeightedMeasure
    m = weightedmeasure(logweight, parent_measure)

    @test @inferred(basemeasure(m)) === parent_measure
    @test @inferred(varshape(m)) == varshape(parent_measure)
    @test @inferred(unshaped(m)) == weightedmeasure(logweight, unshaped(parent_measure))
    @test @inferred(vs(unshaped(m))) == weightedmeasure(logweight, vs(unshaped(parent_measure)))   

    v = rand(parent_dist)
    @test @inferred(BAT.checked_logdensityof(m, v)) == BAT.checked_logdensityof(parent_measure, v) + logweight
    @test @inferred(DensityInterface.logdensityof(m, v)) == DensityInterface.logdensityof(parent_measure, v) + logweight
    @test @inferred(logdensityof(m, v)) == logdensityof(parent_measure, v) + logweight

    rng = bat_rng()
    @test @inferred(rand(deepcopy(rng), BAT.sampler(m), 10)) == rand(deepcopy(rng), BAT.sampler(parent_measure), 10)

    @test cov(unshaped(m)) == cov(unshaped(parent_measure))

    @test @inferred(weightedmeasure(7.9, m)) == renormalize_measure(parent_measure, logweight + 7.9)
end
