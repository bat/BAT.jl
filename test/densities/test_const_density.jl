# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using ArraysOfArrays, Distributions

@testset "const_density" begin
    gen_density_1() = @inferred ConstDensity(
        BAT.HyperRectBounds{Float64}(HyperRectVolume([-1., 0.5], [2.,1]),
        [BAT.hard_bounds, BAT.hard_bounds]),
        one)

    gen_density_n() = @inferred ConstDensity(HyperRectBounds([-1., 0.5], [2.,1],
        hard_bounds), normalize)


    @testset "ConstDensity" begin
        density = gen_density_1()
        @test typeof(density) <: ConstDensity{HyperRectBounds{Float64}, Float64}
        @test density.log_value == zero(Float64)

        density = gen_density_n()
        @test density.log_value ≈ -0.4054651081081644

        pbounds = @inferred param_bounds(density)
        @test pbounds.vol.lo ≈ [-1., 0.5]
        @test nparams(density) == 2
    end

    @testset "convert" begin
        cdensity = @inferred convert(
            AbstractDensity, HyperRectBounds([-1., 0.5], [2.,1], BAT.hard_bounds))
        @test typeof(cdensity) <: ConstDensity
    end

    @testset "density_logval" begin
        density = gen_density_n()
        @test density_logval(density, [1.,2.]) ≈ -0.4054651081081644
    end

    @testset "sampler" begin
        density = gen_density_n()

        s = @inferred sampler(density)
        @test typeof(s) <: SpatialVolume
        @test s.lo ≈ [-1., 0.5]
        @test s.hi ≈ [2., 1.]
    end
end
