# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using ArraysOfArrays, Distributions

@testset "logdval" begin
    @test BAT.density_logval(LinDVal(4.2)) == log(4.2)
    @test BAT.density_logval(LogDVal(log(4.2))) == log(4.2)
end
