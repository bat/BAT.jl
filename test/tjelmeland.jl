# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

@testset "tjelmeland fail" begin
    @test 1 == 2
end

@testset "tjelmeland" begin
    # Computed with pencil and paper
    input = [288, 64, 135] / 487
    κ = 2 # row of interest
    @test transition_matrix2(input, κ) ≈ [359, 64, 0] / 423
end
