# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using Distributions

@testset "const_density" begin
    cd = @inferred ConstDensity(
        BAT.HyperRectBounds{Float64}(HyperRectVolume([-1., 0.5], [2.,1]),
        [BAT.hard_bounds, BAT.hard_bounds]),
        one)

    @testset "ConstDensity" begin
        @test typeof(cd) <: ConstDensity{HyperRectBounds{Float64}, Float64}
        @test cd.log_value == zero(Float64)

        cd = @inferred ConstDensity(HyperRectBounds([-1., 0.5], [2.,1],
            hard_bounds), normalize)
        @test cd.log_value ≈ -0.4054651081081644

        pbounds = @inferred param_bounds(cd)
        @test pbounds.vol.lo ≈ [-1., 0.5]
        @test nparams(cd) == 2
    end

    @testset "convert" begin
        cdensity = @inferred convert(
            AbstractDensity, HyperRectBounds([-1., 0.5], [2.,1], BAT.hard_bounds))
        @test typeof(cdensity) <: ConstDensity
    end

    @testset "unsafe_density_logval" begin
        @test BAT.unsafe_density_logval(cd, [1.,2.]) ≈ -0.4054651081081644
        res = ones(2)
        BAT.unsafe_density_logval!(res, cd, [1. 2.; 3. 4.],
            ExecContext())
        @test res ≈ -0.4054651081081644 * ones(2)
    end

    @testset "ExecCapabilities" begin
        ec = @inferred BAT.exec_capabilities(BAT.unsafe_density_logval, cd)
        @test ec.nthreads == 0
        @test ec.threadsafe == true
        @test ec.nprocs == 0
        @test ec.remotesafe == true

        ec = @inferred BAT.exec_capabilities(BAT.unsafe_density_logval!, ones(2) ,cd)
        @test ec.nthreads == 0
        @test ec.threadsafe == true
        @test ec.nprocs == 0
        @test ec.remotesafe == true
    end

    @testset "sampler" begin
        s = @inferred sampler(cd)
        @test typeof(s) <: SpatialVolume
        @test s.lo ≈ [-1., 0.5]
        @test s.hi ≈ [2., 1.]
    end
end
