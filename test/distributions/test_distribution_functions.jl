# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using Distributions, PDMats, StatsBase


@testset "distribution_functions" begin
    @testset "_check_rand_compat" begin
        @test BAT._check_rand_compat(MvNormal(ones(2)), ones(2,10)) == nothing
        @test_throws DimensionMismatch BAT._check_rand_compat(MvNormal(ones(3)), ones(2,10))
    end

    @testset "BAT.issymmetric_around_origin" begin
        @test BAT.issymmetric_around_origin(Normal(0.0, 3.2)) == true
        @test BAT.issymmetric_around_origin(Normal(1.7, 3.2)) == false
        @test BAT.issymmetric_around_origin(Gamma(4.2, 2.2)) == false
        @test BAT.issymmetric_around_origin(Chisq(20.3)) == false
        @test BAT.issymmetric_around_origin(MvNormal(zeros(2), ones(2))) == true
        @test BAT.issymmetric_around_origin(MvNormal(ones(2), ones(2))) == false
        @test BAT.issymmetric_around_origin(MvTDist(1.5, zeros(2), PDMat(Matrix{Float64}(I, 2, 2))))
        @test BAT.issymmetric_around_origin(MvTDist(1.5, ones(2), PDMat(Matrix{Float64}(I, 2, 2)))) == false
    end
    @testset "BAT.cov2pdmat" begin 
        @test BAT.cov2pdmat(Float64, PDMat(Matrix(1.0I, 2, 2))) == PDMat(Matrix(1.0I, 2, 2))
        @test BAT.cov2pdmat(Float64, PDiagMat([1, 1])) == BAT.cov2pdmat(Float64, Matrix(PDiagMat([1, 1])))
        @test BAT.cov2pdmat(Float64, ScalMat(2, 3)) == BAT.cov2pdmat(Float64, Matrix(ScalMat(2, 3)))
        @test BAT.cov2pdmat(Float64, PDSparseMat(sparse(Matrix(1.0I, 2, 2)))) == BAT.cov2pdmat(Float64, Matrix(PDSparseMat(sparse(Matrix(1.0I, 2, 2)))))
        @test BAT.cov2pdmat(Float64, cholesky(Matrix(1.0I, 2, 2))) == PDMat(cholesky(Matrix(1.0I, 2, 2)))
        @test BAT.cov2pdmat(Float16, cholesky(Matrix(1.0I, 2, 2))) == BAT.cov2pdmat(Float16, Matrix(1.0I, 2, 2))
    end
end
