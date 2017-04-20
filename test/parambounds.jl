# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Base.Test

using IntervalSets


@testset "parameter bounds" begin
    @testset "BAT.apply_bounds" begin
        @test BAT.apply_bounds(+0.3, -1, 2, hard_bounds) ≈ +0.3
        @test BAT.apply_bounds(-0.3, -1, 2, reflective_bounds) ≈ -0.3
        @test BAT.apply_bounds(+1.3, -1, 2, cyclic_bounds) ≈ +1.3

        @test isnan(BAT.apply_bounds(-1.3, -1, 2, hard_bounds))
        @test isnan(BAT.apply_bounds(2.3, -1, 2, hard_bounds))

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
end
