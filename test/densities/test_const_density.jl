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

    @testset "unsafe_density_logval" begin
        density = gen_density_n()
        @test BAT.unsafe_density_logval(density, [1.,2.]) ≈ -0.4054651081081644
        res = ones(2)
        BAT.unsafe_density_logval!(res, density, VectorOfSimilarVectors([1. 2.; 3. 4.]), ExecContext())
        @test res ≈ -0.4054651081081644 * ones(2)
    end

    @testset "ExecCapabilities" begin
        density = gen_density_n()

        ec = @inferred BAT.exec_capabilities(BAT.unsafe_density_logval, density)
        @test ec.nthreads == 0
        @test ec.threadsafe == true
        @test ec.nprocs == 0
        @test ec.remotesafe == true

        ec = @inferred BAT.exec_capabilities(BAT.unsafe_density_logval!, ones(2), density)
        @test ec.nthreads == 0
        @test ec.threadsafe == true
        @test ec.nprocs == 0
        @test ec.remotesafe == true
    end

    @testset "sampler" begin
        density = gen_density_n()

        s = @inferred sampler(density)
        @test typeof(s) <: SpatialVolume
        @test s.lo ≈ [-1., 0.5]
        @test s.hi ≈ [2., 1.]
    end
end
