# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface
using Distributions, Statistics, StatsBase, IntervalSets, ValueShapes

@testset "bat_weighted_measure" begin
    parent_dist = NamedTupleDist(a = Normal(), b = Weibull())
    vs = varshape(parent_dist)
    logweight = 4.2
    parent_density = convert(AbstractMeasureOrDensity, parent_dist)

    @test @inferred(BAT.renormalize_measure(parent_density, logweight)) isa BAT.BATWeightedMeasure
    density = renormalize_measure(parent_density, logweight)

    @test @inferred(parent(density)) === parent_density
    @test @inferred(BAT.measure_support(density)) == BAT.measure_support(parent_density)
    @test @inferred(varshape(density)) == varshape(parent_density)
    @test @inferred(unshaped(density)) == renormalize_measure(unshaped(parent_density), logweight)
    @test @inferred(vs(unshaped(density))) == renormalize_measure(vs(unshaped(parent_density)), logweight)   

    v = rand(parent_dist)
    @test @inferred(BAT.checked_logdensityof(density, v)) == BAT.checked_logdensityof(parent_density, v) + logweight
    @test @inferred(DensityInterface.logdensityof(density, v)) == DensityInterface.logdensityof(parent_density, v) + logweight
    @test @inferred(logdensityof(density, v)) == logdensityof(parent_density, v) + logweight

    rng = bat_rng()
    @test @inferred(rand(deepcopy(rng), BAT.sampler(density), 10)) == rand(deepcopy(rng), BAT.sampler(parent_density), 10)

    @test cov(unshaped(density)) == cov(unshaped(parent_density))

    @test @inferred(weightedmeasure(7.9, density)) == renormalize_measure(parent_density, logweight + 7.9)
end
