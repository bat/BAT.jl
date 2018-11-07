# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using Distributions, PDMats, StatsBase


@testset "distribution_functions" begin
    @testset "_iszero" begin
        @test BAT._iszero(zero(Float64))
        @test BAT._iszero(zeros(Float32,4,4))
        @test !BAT._iszero(ones(Float32,4,4))
        @test BAT._iszero(Distributions.ZeroVector(Float64, 4))
    end

    @testset "_check_rand_compat" begin
        @test BAT._check_rand_compat(MvNormal(ones(2)), ones(2,10)) == nothing
        @test_throws DimensionMismatch BAT._check_rand_compat(MvNormal(ones(3)), ones(2,10))
    end

    @testset "issymmetric_around_origin" begin
        @test issymmetric_around_origin(Normal(0.0, 3.2)) == true
        @test issymmetric_around_origin(Normal(1.7, 3.2)) == false
        @test issymmetric_around_origin(Gamma(4.2, 2.2)) == false
        @test issymmetric_around_origin(Chisq(20.3)) == false
        @test issymmetric_around_origin(MvNormal(zeros(2), ones(2))) == true
        @test issymmetric_around_origin(MvNormal(ones(2), ones(2))) == false
        @test issymmetric_around_origin(MvTDist(1.5, zeros(2), PDMat(Matrix{Float64}(I, 2, 2))))
        @test issymmetric_around_origin(MvTDist(1.5, ones(2), PDMat(Matrix{Float64}(I, 2, 2)))) == false
    end
end
