# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using ArraysOfArrays, Distributions

@testset "logdval" begin
    @test logvalof(LinDVal(4.2)) == log(4.2)
    @test logvalof(LogDVal(log(4.2))) == log(4.2)
end
