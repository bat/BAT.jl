# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions

@testset "shims" begin
    @testset "_iszero" begin
        @test BAT._iszero(zero(Float64))
        @test BAT._iszero(zeros(Float32,4,4))
        @test !BAT._iszero(ones(Float32,4,4))
        @test BAT._iszero(Distributions.ZeroVector(Float64, 4))
    end
end

