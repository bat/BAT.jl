# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions

using BAT: convolution, gaussian_kernel

@testset "convolution_utils" begin
    input = [[3,2,4,2,7,6] [8,0,2,1,7,8] [2,2,10,4,1,9] [1,5,4,6,5,0] [5,4,1,7,5,6] [5,0,2,7,6,8]]
    filter = [[1,1,1] [0,0,0] [-1,-1,-1]]
    # Expected result computed independently:
    expected = [[-5,-8,-2,1] [0,-12,-5,5] [4,4,2,-4] [3,6,0,-10]]

    @testset "2D convolution" begin
        padded = @inferred(convolution(input, filter))
        unpadded = @inferred(convolution(input, filter, padding = :none))

        @test unpadded isa Matrix{<:AbstractFloat}
        @test size(unpadded) == (4, 4)
        @test unpadded == expected

        @test padded isa Matrix{<:AbstractFloat}
        @test size(padded) == size(input)
        @test padded[2:5, 2:5] == unpadded
    end

    @testset "Gaussian kernel" begin
        g = pdf.(Normal(), -2:2)
        k = g / sum(g)
        ref_kernel = k * k'

        kernel_1d = @inferred(gaussian_kernel(1.0, l = 5))
        @test kernel_1d isa Matrix{<:AbstractFloat}
        @test size(kernel_1d) == (5, 5)
        @test kernel_1d ≈ ref_kernel

        kernel_2d = @inferred(gaussian_kernel((1.0, 1.0), l = (5, 5)))
        @test kernel_2d isa Matrix{<:AbstractFloat}
        @test size(kernel_2d) == (5, 5)
        @test kernel_2d ≈ ref_kernel
    end
end
