# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using ArraysOfArrays, Distributions

@testset "logdval" begin
    @test @inferred(logvalof(LogDVal(log(4.2)))) == log(4.2)
    @test @inferred(logvalof((foo = 55, log = log(4.2), bar = 2.7))) == log(4.2)
    @test_throws ArgumentError logvalof(42)
end
