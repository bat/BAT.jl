# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using ArraysOfArrays, IntervalSets

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
        @test BAT.isoob([1.0, BAT.oob(Float64), 3.0]) == true
        @test BAT.isoob([1.0, 2.0, 3.0]) == false
    end

    @testset "BAT.apply_bounds" begin
        @test BAT.apply_bounds(+0.3, -1, 2, BAT.hard_bounds) ≈ +0.3
        @test BAT.apply_bounds(-0.3, -1, 2, BAT.reflective_bounds) ≈ -0.3
        @test BAT.apply_bounds(+1.3, -1, 2, BAT.cyclic_bounds) ≈ +1.3

        @test isnan(BAT.apply_bounds(-1.3, -1, 2, BAT.hard_bounds))
        @test isnan(BAT.apply_bounds(2.3, -1, 2, BAT.hard_bounds))
        @test BAT.apply_bounds(-1.3, -1, 2, BAT.hard_bounds, 5) ≈ 5
        @test BAT.apply_bounds(2.3, -1, 2, BAT.hard_bounds, 5) ≈ 5

        @test BAT.apply_bounds(-1.3, -1, 2, BAT.reflective_bounds) ≈ -0.7
        @test BAT.apply_bounds(-4.3, -1, 2, BAT.reflective_bounds) ≈ +1.7
        @test BAT.apply_bounds(-7.3, -1, 2, BAT.reflective_bounds) ≈ -0.7
        @test BAT.apply_bounds(+2.3, -1, 2, BAT.reflective_bounds) ≈ +1.7
        @test BAT.apply_bounds(+5.3, -1, 2, BAT.reflective_bounds) ≈ -0.7
        @test BAT.apply_bounds(+8.3, -1, 2, BAT.reflective_bounds) ≈ +1.7

        @test BAT.apply_bounds(-1.3, -1, 2, BAT.cyclic_bounds) ≈ +1.7
        @test BAT.apply_bounds(-4.3, -1, 2, BAT.cyclic_bounds) ≈ +1.7
        @test BAT.apply_bounds(-7.3, -1, 2, BAT.cyclic_bounds) ≈ +1.7
        @test BAT.apply_bounds(+2.3, -1, 2, BAT.cyclic_bounds) ≈ -0.7
        @test BAT.apply_bounds(+5.3, -1, 2, BAT.cyclic_bounds) ≈ -0.7
        @test BAT.apply_bounds(+8.3, -1, 2, BAT.cyclic_bounds) ≈ -0.7

        @test typeof(@inferred BAT.apply_bounds(+0.3, -1, 2, BAT.hard_bounds)) == Float64
        @test typeof(@inferred BAT.apply_bounds(+0.3f0, -1, 2, BAT.hard_bounds)) == Float32
        @test typeof(@inferred BAT.apply_bounds(+0.3f0, -1, 2.0, BAT.hard_bounds)) == Float64

        @test BAT.apply_bounds(+5.3, ClosedInterval(-1, 2), BAT.reflective_bounds) ≈ -0.7
    end

    @testset "BAT.NoParamBounds" begin
        n = 2
        @test typeof(@inferred BAT.NoParamBounds(n)) == BAT.NoParamBounds

        params = [-1000., 1000]
        uparams = BAT.NoParamBounds(n)
        @test nparams(uparams) == n
        @test params in uparams
        @test in( VectorOfSimilarVectors(hcat(params, params)), uparams, 1)

        @test BAT.apply_bounds!(params, uparams) == params
    end

    @testset "BAT.HyperRectBounds" begin
        @test typeof(@inferred BAT.HyperRectBounds([-1., 0.5], [2.,1], BAT.BAT.hard_bounds)) <: BAT.ParamVolumeBounds{Float64, BAT.HyperRectVolume{Float64}}
        @test_throws ArgumentError BAT.HyperRectBounds([-1.], [2.,1], [BAT.hard_bounds, BAT.reflective_bounds])
        @test_throws ArgumentError BAT.HyperRectBounds([-1., 2.], [2.,1], [BAT.hard_bounds, BAT.reflective_bounds])

        hyperRectBounds = @inferred BAT.HyperRectBounds([-1., -1.], [0.5,1], [BAT.hard_bounds, BAT.hard_bounds])
        @test nparams(hyperRectBounds) == 2
        @test [0.0, 0.0] in hyperRectBounds
        @test ([0.5, 2] in hyperRectBounds) == false

        hyperRectBounds = @inferred BAT.HyperRectBounds([ClosedInterval(-1.,0.5),
            ClosedInterval(-1.,1.)], [BAT.hard_bounds, BAT.hard_bounds])
        @test nparams(hyperRectBounds) == 2
        @test [0.0, 0.0] in hyperRectBounds
        @test ([0.5, 2] in hyperRectBounds) == false

        @test BAT.apply_bounds!([0.3, -4.3, -7.3], BAT.HyperRectBounds([-1.,-1,-1], [2.,2,2],
            [BAT.hard_bounds, BAT.reflective_bounds, BAT.cyclic_bounds])) ≈ [+0.3, 1.7, 1.7]

        @test BAT.apply_bounds!(
            VectorOfSimilarVectors([0.3 0.3 0.3; 0.3 -7.3 +8.3; 0.3 -7.3 +8.3]),
            BAT.HyperRectBounds([-1., -1., -1], [2., 2.,2.], [BAT.hard_bounds, BAT.reflective_bounds, BAT.cyclic_bounds])
        ) ≈ VectorOfSimilarVectors([+0.3 0.3 0.3; 0.3 -0.7 1.7; 0.3 1.7 -0.7])

        @test BAT.isoob(BAT.apply_bounds!(rand!(MersenneTwister(7002), zeros(Float64, 2)), hyperRectBounds))
        @test BAT.isoob(BAT.apply_bounds!(rand!(MersenneTwister(7002),
            BAT.spatialvolume(hyperRectBounds), zeros(Float64, 2)), hyperRectBounds)) == false

        @test BAT.isoob(BAT.apply_bounds!([0,2.],hyperRectBounds))
        @test BAT.apply_bounds!([0,2.],hyperRectBounds,false) ≈ [0,2.]
    end

    @testset "similar" begin
        a = @inferred BAT.HyperRectBounds([-1., 0.5], [2.,1], BAT.BAT.hard_bounds)
        c = @inferred similar(a)
        @test typeof(c) <:BAT.HyperRectBounds{Float64}
        @test typeof(c.vol) <: BAT.HyperRectVolume{Float64}
        @test typeof(c.bt) <: Array{BAT.BoundsType,1}
    end

    @testset "BAT.intersect" begin
        a = @inferred apb_test(3)
        b = @inferred apb_test(4)
        @test_throws ArgumentError intersect(a,b)
        b = @inferred apb_test(3)
        @test intersect(a,b)
        b = @inferred BAT.NoParamBounds(3)
        @test BAT.unsafe_intersect(b, BAT.NoParamBounds(3)) == b
        @test BAT.unsafe_intersect(a, BAT.NoParamBounds(3)) == a
        @test BAT.unsafe_intersect(BAT.NoParamBounds(3), a) == a

        hb = BAT.hard_bounds
        rb = BAT.reflective_bounds
        cb = BAT.cyclic_bounds

        @test intersect(hb, rb) == BAT.hard_bounds
        @test intersect(cb, hb) == BAT.hard_bounds
        @test intersect(rb, cb) == BAT.reflective_bounds
        @test intersect(cb, rb) == BAT.reflective_bounds
        @test intersect(cb, cb) == BAT.cyclic_bounds

        hyperRectBounds_a = @inferred BAT.HyperRectBounds([-1., -1.], [2.,3.],
            [BAT.hard_bounds, BAT.cyclic_bounds])
        hyperRectBounds_b = @inferred BAT.HyperRectBounds([-0.5, -0.5], [1.,2.],
            [BAT.cyclic_bounds, BAT.cyclic_bounds])

        res = @inferred intersect(hyperRectBounds_a, hyperRectBounds_b)
        @test res.vol.lo ≈ hyperRectBounds_b.vol.lo
        @test res.vol.hi ≈ hyperRectBounds_b.vol.hi
        @test res.bt == hyperRectBounds_b.bt

        res = @inferred intersect(hyperRectBounds_b, hyperRectBounds_a)
        @test res.vol.lo ≈ hyperRectBounds_b.vol.lo
        @test res.vol.hi ≈ hyperRectBounds_b.vol.hi
        @test res.bt == hyperRectBounds_b.bt

        hyperRectBounds_c = @inferred BAT.HyperRectBounds([-0.5, -0.5], [4.,4.],
            [BAT.cyclic_bounds, BAT.reflective_bounds])

        res = @inferred intersect(hyperRectBounds_a, hyperRectBounds_c)
        @test res.vol.lo ≈ hyperRectBounds_c.vol.lo
        @test res.vol.hi ≈ hyperRectBounds_a.vol.hi
        @test res.bt == [BAT.hard_bounds, BAT.reflective_bounds]

    end
end
