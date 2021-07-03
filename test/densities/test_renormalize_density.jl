# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, Statistics, StatsBase, IntervalSets, ValueShapes

@testset "truncated_density" begin
    parent_dist = NamedTupleDist(a = Normal(), b = Weibull())
    vs = varshape(parent_dist)
    logrenormf = 4.2
    parent_density = convert(AbstractDensity, parent_dist)

    @test @inferred(BAT.renormalize_density(parent_density, logrenormf)) isa BAT.RenormalizedDensity
    density = renormalize_density(parent_density, logrenormf)

    @test @inferred(parent(density)) === parent_density
    @test @inferred(BAT.var_bounds(density)) == BAT.var_bounds(parent_density)
    @test @inferred(varshape(density)) == varshape(parent_density)
    @test @inferred(unshaped(density)) == renormalize_density(unshaped(parent_density), logrenormf)
    @test @inferred(vs(unshaped(density))) == renormalize_density(vs(unshaped(parent_density)), logrenormf)   

    v = rand(parent_dist)
    @test @inferred(BAT.eval_logval(density, v, Float64)) == BAT.eval_logval(parent_density, v, Float64) + logrenormf
    @test @inferred(BAT.eval_logval_unchecked(density, v)) == BAT.eval_logval_unchecked(parent_density, v) + logrenormf
    @test @inferred(logdensityof(density, v)) == logdensityof(parent_density, v) + logrenormf

    rng = bat_rng()
    @test @inferred(rand(deepcopy(rng), BAT.sampler(density), 10)) == rand(deepcopy(rng), BAT.sampler(parent_density), 10)
    @test @inferred(rand(deepcopy(rng), BAT.bat_sampler(density), 10)) == rand(deepcopy(rng), BAT.bat_sampler(parent_density), 10)

    @test cov(unshaped(density)) == cov(unshaped(parent_density))

    @test @inferred(renormalize_density(density, 7.9)) == renormalize_density(parent_density, logrenormf + 7.9)
end
