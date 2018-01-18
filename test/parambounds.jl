# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

using IntervalSets

struct apb_test <: BAT.AbstractParamBounds
    nparams::Integer
end

BAT.nparams(a::apb_test) = a.nparams
BAT.unsafe_intersect(a::apb_test, b::apb_test) = true

@testset "parameter bounds" begin
    @testset "BAT.oob" begin
        @test BAT.isoob(BAT.oob(1.))
        @test BAT.isoob(1.) == false
        @test BAT.isoob(BAT.oob(1))
        @test BAT.isoob(1) == false
        @test BAT.isoob(BAT.oob(ones(Float64,2,2)))
        @test BAT.isoob(ones(Float64,2,2)) == false
    end

    @testset "BAT.apply_bounds" begin
        @test BAT.apply_bounds(+0.3, -1, 2, hard_bounds) ≈ +0.3
        @test BAT.apply_bounds(-0.3, -1, 2, reflective_bounds) ≈ -0.3
        @test BAT.apply_bounds(+1.3, -1, 2, cyclic_bounds) ≈ +1.3

        @test isnan(BAT.apply_bounds(-1.3, -1, 2, hard_bounds))
        @test isnan(BAT.apply_bounds(2.3, -1, 2, hard_bounds))
        @test BAT.apply_bounds(-1.3, -1, 2, hard_bounds, 5) ≈ 5
        @test BAT.apply_bounds(2.3, -1, 2, hard_bounds, 5) ≈ 5

        @test BAT.apply_bounds(-1.3, -1, 2, reflective_bounds) ≈ -0.7
        @test BAT.apply_bounds(-4.3, -1, 2, reflective_bounds) ≈ +1.7
        @test BAT.apply_bounds(-7.3, -1, 2, reflective_bounds) ≈ -0.7
        @test BAT.apply_bounds(+2.3, -1, 2, reflective_bounds) ≈ +1.7
        @test BAT.apply_bounds(+5.3, -1, 2, reflective_bounds) ≈ -0.7
        @test BAT.apply_bounds(+8.3, -1, 2, reflective_bounds) ≈ +1.7

        @test BAT.apply_bounds(-1.3, -1, 2, cyclic_bounds) ≈ +1.7
        @test BAT.apply_bounds(-4.3, -1, 2, cyclic_bounds) ≈ +1.7
        @test BAT.apply_bounds(-7.3, -1, 2, cyclic_bounds) ≈ +1.7
        @test BAT.apply_bounds(+2.3, -1, 2, cyclic_bounds) ≈ -0.7
        @test BAT.apply_bounds(+5.3, -1, 2, cyclic_bounds) ≈ -0.7
        @test BAT.apply_bounds(+8.3, -1, 2, cyclic_bounds) ≈ -0.7

        @test typeof(@inferred BAT.apply_bounds(+0.3, -1, 2, hard_bounds)) == Float64
        @test typeof(@inferred BAT.apply_bounds(+0.3f0, -1, 2, hard_bounds)) == Float32
        @test typeof(@inferred BAT.apply_bounds(+0.3f0, -1, 2.0, hard_bounds)) == Float64

        @test BAT.apply_bounds(+5.3, ClosedInterval(-1, 2), reflective_bounds) ≈ -0.7
    end

    @testset "BAT.NoParamBounds" begin
        n = 2
        @test typeof(@inferred BAT.NoParamBounds(n)) == BAT.NoParamBounds

        params = [-1000., 1000]
        uparams = BAT.NoParamBounds(n)
        @test nparams(uparams) == n
        @test params in uparams
        @test in( hcat(params, params), uparams, 1)

        @test BAT.apply_bounds!(params, uparams) == params
    end

    @testset "BAT.HyperRectBounds" begin
        @test typeof(@inferred BAT.HyperRectBounds([-1., 0.5], [2.,1], BAT.hard_bounds)) <: ParamVolumeBounds{Float64, HyperRectVolume{Float64}}
        @test_throws ArgumentError BAT.HyperRectBounds([-1.], [2.,1], [hard_bounds, reflective_bounds])
        #@test_throws ArgumentError BAT.HyperRectBounds([-1., 2.], [2.,1], [hard_bounds, reflective_bounds])

        hyperRectBounds = BAT.HyperRectBounds([-1., -1.], [0.5,1], [hard_bounds, hard_bounds])
        @test nparams(hyperRectBounds) == 2
        @test [0.0, 0.0] in hyperRectBounds
        @test ([0.5, 2] in hyperRectBounds) == false

        @test BAT.apply_bounds!([0.3, -4.3, -7.3], BAT.HyperRectBounds([-1.,-1,-1], [2.,2,2], [hard_bounds, reflective_bounds, cyclic_bounds])) ≈ [+0.3, 1.7, 1.7]

        @test BAT.apply_bounds!([0.3 0.3 0.3; 0.3 -7.3 +8.3; 0.3 -7.3 +8.3], BAT.HyperRectBounds([-1., -1., -1], [2., 2.,2.], [hard_bounds, reflective_bounds, cyclic_bounds])) ≈ [+0.3 0.3 0.3;0.3 -0.7 1.7;0.3 1.7 -0.7]

        @test BAT.isoob(BAT.apply_bounds!(rand!(MersenneTwister(7002), zeros(Float64, 2, 2)), hyperRectBounds))
        @test BAT.isoob(BAT.apply_bounds!(rand!(MersenneTwister(7002), BAT.spatialvolume(hyperRectBounds), zeros(Float64, 2, 2)), hyperRectBounds)) == false
    end

    @testset "similar" begin
        a = @inferred BAT.HyperRectBounds([-1., 0.5], [2.,1], BAT.hard_bounds)
        c = @inferred similar(a)
        @test typeof(c) <:HyperRectBounds{Float64}
        @test typeof(c.vol) <: BAT.HyperRectVolume{Float64}
        @test typeof(c.bt) <: Array{BoundsType,1}
    end

    @testset "BAT.intersect" begin
        a = @inferred apb_test(3)
        b = @inferred apb_test(4)
        @test_throws ArgumentError intersect(a,b)
        b = @inferred apb_test(3)
        @test intersect(a,b)
        b = @inferred NoParamBounds(3)
        @test BAT.unsafe_intersect(b, NoParamBounds(3)) == b
        @test BAT.unsafe_intersect(a, NoParamBounds(3)) == a
        @test BAT.unsafe_intersect(NoParamBounds(3), a) == a
    end
end
