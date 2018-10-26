# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using IntervalSets

@testset "util" begin
    @testset "_car_cdr" begin
        @test BAT._car_cdr_impl() == ()
        @test BAT._car_cdr_impl(4, 3.5, -1) == (4, (3.5, -1))
        @test BAT._car_cdr((3, -8, 2)) == (3, (-8, 2))
    end

    @testset "_all_lteq" begin
        @test BAT._all_lteq(-0.4, -0.4:0.4:0.8 ,0.8)
        @test !BAT._all_lteq(-0.4, -0.4:0.4:0.8 ,0.7)
        @test !BAT._all_lteq(-0.3, -0.4:0.4:0.8 ,0.8)

        @test BAT._all_lteq([0.0, 0.5, -0.4], [0.3, 0.5, -0.4] , [0.3, 0.6, -0.3])
        @test !BAT._all_lteq([0.5, -0.4], [0.49, -0.4] , [0.6, -0.3])
        @test_throws DimensionMismatch BAT._all_lteq(
            [0.5, -0.4], [0.3, 0.5, -0.4] , [0.3, 0.6, -0.3])

        @test BAT._all_in_ui(ones(2, 2))
        @test !BAT._all_in_ui([0.4, 0.2, 1.1])
    end

    @testset "fromui" begin
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
