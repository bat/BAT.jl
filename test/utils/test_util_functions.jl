# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test


@testset "util_functions" begin
    @test BAT.choose_something(42, 47) === 42
    @test BAT.choose_something(nothing, 47) === 47
    @test BAT.choose_something(missing, missing) === missing
end
