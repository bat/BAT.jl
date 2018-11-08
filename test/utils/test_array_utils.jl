# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test


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
    end
end
