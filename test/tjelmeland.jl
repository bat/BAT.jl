# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

@testset "tjelmeland" begin
    # Computed with pencil and paper
    input = [288, 64, 135] / 487
    κ = 3 # row of interest
    @test multipropT2(input, κ) ≈ [359, 64, 0] / 423

    input2 = [0.2, 0.5, 0.3]
    k2 = 3
    @test multipropT2(input2, k2) ≈ [0.25, 0.75, 0.0]

    input3 = [324, 3, 31, 123, 12]
    k3 = 3
    output = multipropT2(input3, k3)
    @test sum(output) ≈ 1.

    # invalid input should lead to errors
    @test_throws ArgumentError multipropT2(input, -2)
    @test_throws ArgumentError multipropT2(-input, 2)
end
