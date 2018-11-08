# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using IntervalSets

@testset "util" begin
    @testset "fromui" begin
        @test BAT._all_in_ui(ones(2, 2))
        @test !BAT._all_in_ui([0.4, 0.2, 1.1])

        @test fromui(0.5, -1.0, 1.0) ≈ 0.0
        @test fromui(0.5, -0.5, 5.0) ≈ 2.25
        @test fromui(0.1, -0.5, 1.0) ≈ -0.35
        @test_throws ArgumentError fromui(-0.1, -1.0, 1.0)
        @test_throws ArgumentError fromui(1.1, -0.5, 1.0)

        @test fromui(0.5, ClosedInterval(-0.5, 1.0)) ≈ 0.25

        inv_fromui = @inferred inv(fromui)
        @test inv_fromui  == BAT.inv_fromui
        @test inv(inv_fromui) == fromui

        @test inv_fromui(2.25, -0.5, 5.0) ≈ 0.5

        @test inv_fromui(-0.35, -0.5, 1.0) ≈ 0.1
        @test inv_fromui(5.0, -0.5, 5.0) ≈ 1.0
        @test inv_fromui(-0.5, -0.5, 5.0) ≈ 0.0
        @test_throws ArgumentError inv_fromui(-0.2, -0.1, 3)
        @test_throws ArgumentError inv_fromui(3.1, -0.1, 3)

        @test inv_fromui(-0.35, ClosedInterval(-0.5, 1.0)) ≈ 0.1
    end
end
