# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using ArraysOfArrays, ValueShapes, Distributions

@testset "const_density" begin
    gen_density_1() =  BAT.ConstDensity(LogDVal(0), BAT.HyperRectBounds{Float64}(BAT.HyperRectVolume([-1., 0.5], [2.,1])))

    gen_density_n() = BAT.ConstDensity(normalize, BAT.HyperRectBounds([-1., 0.5], [2.,1]))

    @testset "BAT.ConstDensity" begin
        density = @inferred gen_density_1()
        @test typeof(density) == ConstDensity{Int,BAT.HyperRectBounds{Float64}}
        @test density.value == LogDVal(zero(Int))

        density = @inferred gen_density_n()
        @test density.value.logval ≈ -0.4054651081081644

        pbounds = @inferred BAT.var_bounds(density)
        @test pbounds.vol.lo ≈ [-1., 0.5]
        @test totalndof(density) == 2
    end

    @testset "convert" begin
        cdensity = @inferred BAT.ConstDensity(normalize, BAT.HyperRectBounds([-1., 0.5], [2.,1]))
        @test typeof(cdensity) <: BAT.ConstDensity
    end

    @testset "DensityInterface.logdensityof" begin
        density = gen_density_n()
        @test DensityInterface.logdensityof(density, [1.,2.]) ≈ -0.4054651081081644
    end

    @testset "sampler" begin
        density = gen_density_n()

        s = @inferred BAT.bat_sampler(density)
        @test typeof(s) <: BAT.SpatialVolume
        @test s.lo ≈ [-1., 0.5]
        @test s.hi ≈ [2., 1.]
    end
end
