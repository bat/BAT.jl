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
   @testset "BAT.NoVarBounds" begin
        n = 2
        @test typeof(@inferred BAT.NoVarBounds(n)) == BAT.NoVarBounds

        v = [-1000., 1000]
        nobounds = BAT.NoVarBounds(n)
        @test @inferred(totalndof(nobounds)) == n
        @test @inferred(v in nobounds)
    end

    @testset "BAT.HyperRectBounds" begin
        @test typeof(@inferred BAT.HyperRectBounds([-1., 0.5], [2.,1])) <: BAT.VarVolumeBounds{Float64, BAT.HyperRectVolume{Float64}}
        @test_throws ArgumentError BAT.HyperRectBounds([-1.], [2.,1])
        @test_throws ArgumentError BAT.HyperRectBounds([-1., 2.], [2.,1])

        hyperRectBounds = @inferred BAT.HyperRectBounds([-1., -1.], [0.5,1])
        @test totalndof(hyperRectBounds) == 2
        @test [0.0, 0.0] in hyperRectBounds
        @test ([0.5, 2] in hyperRectBounds) == false

        hyperRectBounds = @inferred BAT.HyperRectBounds([ClosedInterval(-1.,0.5), ClosedInterval(-1.,1.)])
        @test totalndof(hyperRectBounds) == 2
        @test [0.0, 0.0] in hyperRectBounds
        @test ([0.5, 2] in hyperRectBounds) == false
    end

    @testset "similar" begin
        a = @inferred BAT.HyperRectBounds([-1., 0.5], [2.,1])
        c = @inferred similar(a)
        @test typeof(c) <:BAT.HyperRectBounds{Float64}
        @test typeof(c.vol) <: BAT.HyperRectVolume{Float64}
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

        hyperRectBounds_a = @inferred BAT.HyperRectBounds([-1., -1.], [2.,3.])
        hyperRectBounds_b = @inferred BAT.HyperRectBounds([-0.5, -0.5], [1.,2.])

        res = @inferred intersect(hyperRectBounds_a, hyperRectBounds_b)
        @test res.vol.lo ≈ hyperRectBounds_b.vol.lo
        @test res.vol.hi ≈ hyperRectBounds_b.vol.hi

        res = @inferred intersect(hyperRectBounds_b, hyperRectBounds_a)
        @test res.vol.lo ≈ hyperRectBounds_b.vol.lo
        @test res.vol.hi ≈ hyperRectBounds_b.vol.hi

        hyperRectBounds_c = @inferred BAT.HyperRectBounds([-0.5, -0.5], [4.,4.])

        res = @inferred intersect(hyperRectBounds_a, hyperRectBounds_c)
        @test res.vol.lo ≈ hyperRectBounds_c.vol.lo
        @test res.vol.hi ≈ hyperRectBounds_a.vol.hi
    end
end
