# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test


@testset "util_functions" begin
    @test @inferred(BAT.choose_something(42, 47)) === 42
    @test @inferred(BAT.choose_something(nothing, 47)) === 47
    @test @inferred(BAT.choose_something(nothing, nothing)) === nothing
    @test @inferred(BAT.choose_something(missing, 47)) === 47
    @test @inferred(BAT.choose_something(missing, missing)) === missing
    @test_throws ArgumentError BAT.choose_something(missing, nothing)
    @test_throws ArgumentError BAT.choose_something(nothing, missing)
end
