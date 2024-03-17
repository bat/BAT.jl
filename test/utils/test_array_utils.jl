# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test


@testset "util" begin
    @testset "BAT._car_cdr" begin
        @test BAT._car_cdr_impl() == ()
        @test BAT._car_cdr_impl(4, 3.5, -1) == (4, (3.5, -1))
        @test BAT._car_cdr((3, -8, 2)) == (3, (-8, 2))
    end
end
