# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using ArraysOfArrays, IntervalSets, ValueShapes

struct apb_test <: BAT.AbstractVarBounds
    varndof::Integer
end

ValueShapes.totalndof(a::apb_test) = a.varndof
BAT.unsafe_intersect(a::apb_test, b::apb_test) = true

@testset "parameter bounds" begin
    @testset "BAT.renormalize_variate_impl" begin
        @test BAT.renormalize_variate_impl(+0.3, -1, 2, BAT.hard_bounds) ≈ +0.3
        @test BAT.renormalize_variate_impl(-0.3, -1, 2, BAT.reflective_bounds) ≈ -0.3
        @test BAT.renormalize_variate_impl(+1.3, -1, 2, BAT.cyclic_bounds) ≈ +1.3

        @test BAT.renormalize_variate_impl(-1.3, -1, 2, BAT.hard_bounds) == -1.3
        @test BAT.renormalize_variate_impl(2.3, -1, 2, BAT.hard_bounds) == 2.3

        @test BAT.renormalize_variate_impl(-1.3, -1, 2, BAT.reflective_bounds) ≈ -0.7
        @test BAT.renormalize_variate_impl(-4.3, -1, 2, BAT.reflective_bounds) ≈ +1.7
        @test BAT.renormalize_variate_impl(-7.3, -1, 2, BAT.reflective_bounds) ≈ -0.7
        @test BAT.renormalize_variate_impl(+2.3, -1, 2, BAT.reflective_bounds) ≈ +1.7
        @test BAT.renormalize_variate_impl(+5.3, -1, 2, BAT.reflective_bounds) ≈ -0.7
        @test BAT.renormalize_variate_impl(+8.3, -1, 2, BAT.reflective_bounds) ≈ +1.7

        @test BAT.renormalize_variate_impl(-1.3, -1, 2, BAT.cyclic_bounds) ≈ +1.7
        @test BAT.renormalize_variate_impl(-4.3, -1, 2, BAT.cyclic_bounds) ≈ +1.7
        @test BAT.renormalize_variate_impl(-7.3, -1, 2, BAT.cyclic_bounds) ≈ +1.7
        @test BAT.renormalize_variate_impl(+2.3, -1, 2, BAT.cyclic_bounds) ≈ -0.7
        @test BAT.renormalize_variate_impl(+5.3, -1, 2, BAT.cyclic_bounds) ≈ -0.7
        @test BAT.renormalize_variate_impl(+8.3, -1, 2, BAT.cyclic_bounds) ≈ -0.7
    
        @test typeof(@inferred BAT.renormalize_variate_impl(+0.3, -1, 2, BAT.hard_bounds)) == Float64
        @test typeof(@inferred BAT.renormalize_variate_impl(+0.3f0, -1, 2, BAT.hard_bounds)) == Float32
        @test typeof(@inferred BAT.renormalize_variate_impl(+0.3f0, -1, 2.0, BAT.hard_bounds)) == Float64

        @test BAT.renormalize_variate_impl(+5.3, ClosedInterval(-1, 2), BAT.reflective_bounds) ≈ -0.7
    end

    @testset "BAT.NoVarBounds" begin
        n = 2
        @test typeof(@inferred BAT.NoVarBounds(n)) == BAT.NoVarBounds

        v = [-1000., 1000]
        nobounds = BAT.NoVarBounds(n)
        @test @inferred(totalndof(nobounds)) == n
        @test @inferred(v in nobounds)
        @test @inferred(BAT.renormalize_variate(nobounds, v)) === v
    end

    @testset "BAT.HyperRectBounds" begin
        @test typeof(@inferred BAT.HyperRectBounds([-1., 0.5], [2.,1], BAT.BAT.hard_bounds)) <: BAT.VarVolumeBounds{Float64, BAT.HyperRectVolume{Float64}}
        @test_throws ArgumentError BAT.HyperRectBounds([-1.], [2.,1], [BAT.hard_bounds, BAT.reflective_bounds])
        @test_throws ArgumentError BAT.HyperRectBounds([-1., 2.], [2.,1], [BAT.hard_bounds, BAT.reflective_bounds])

        hyperRectBounds = @inferred BAT.HyperRectBounds([-1., -1.], [0.5,1], [BAT.hard_bounds, BAT.hard_bounds])
        @test totalndof(hyperRectBounds) == 2
        @test [0.0, 0.0] in hyperRectBounds
        @test ([0.5, 2] in hyperRectBounds) == false

        hyperRectBounds = @inferred BAT.HyperRectBounds([ClosedInterval(-1.,0.5),
            ClosedInterval(-1.,1.)], [BAT.hard_bounds, BAT.hard_bounds])
        @test totalndof(hyperRectBounds) == 2
        @test [0.0, 0.0] in hyperRectBounds
        @test ([0.5, 2] in hyperRectBounds) == false

        @test @inferred(BAT.renormalize_variate(
            BAT.HyperRectBounds([-1.,-1,-1], [2.,2,2], [BAT.hard_bounds, BAT.reflective_bounds, BAT.cyclic_bounds]),
            [0.3, -4.3, -7.3]
        )) ≈ [+0.3, 1.7, 1.7]

        @test BAT.renormalize_variate(hyperRectBounds, [0,2.]) == [0,2.]
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
        b = @inferred BAT.NoVarBounds(3)
        @test BAT.unsafe_intersect(b, BAT.NoVarBounds(3)) == b
        @test BAT.unsafe_intersect(a, BAT.NoVarBounds(3)) == a
        @test BAT.unsafe_intersect(BAT.NoVarBounds(3), a) == a

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
