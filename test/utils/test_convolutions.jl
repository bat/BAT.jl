# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
using Distributions

#2d Convolution
input = [[3,2,4,2,7,6] [8,0,2,1,7,8] [2,2,10,4,1,9] [1,5,4,6,5,0] [5,4,1,7,5,6] [5,0,2,7,6,8]]
filter = [[1,1,1] [0,0,0] [-1,-1,-1]]
output = [[-5,-8,-2,1] [0,-12,-5,5] [4,4,2,-4] [3,6,0,-10]]#Output calculated in another way

#Gaussian Kernel
gaussian_1d = [pdf(Normal(), x) for x in -4:4]
gaussian_kernel_1d = g*g'
kernel_test_1d = (gaussian_kernel(1., l = 5))

@testset "Convolution utilities" begin
    @testset "2D Convolution" begin
        padded = @inferred(convolution(input, filter))
        no_padded = @inferred(convolution(input, filter, padding = "No"))

        #No-Padded tests
        @test no_padded isa Matrix{<:AbstractFloat}
        @test size(no_padded) == (4,4)
        @test all(x-> x == 0, no_padded - output)#Check that result is equal to preduction

        #Padded tests
        @test padded isa Matrix{<:AbstractFloat}
        @test size(padded) == size(input)
        @test all(x-> x == 0, padded[2:5,2:5] - no_padded)#Check that result is equal to no_padded in same range
    end
    @testset "Gaussian Kernel" begin
        kernel_test_1d = @inferred(gaussian_kernel(1., l = 5)) #1d
        kernel_test_2d = @inferred(gaussian_kernel((1., 1.), l = (5, 5))) #2d
        
        @test kernel_test_1d isa Matrix{<:AbstractFloat}
        @test size(kernel_test_1d) == (5,5)
        @test all(x -> x <1e-2 ,  kernel_test_1d - gaussian_kernel_1d)

        @test kernel_test_2d isa Matrix{<:AbstractFloat}
        @test size(kernel_test_2d) == (5,5)
        @test all(x -> x < 1e-2 ,  kernel_test_2d - gaussian_kernel_1d)
    end
end