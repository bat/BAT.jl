# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Distributions, Test


@testset "Student-t affine transform initialization" begin
    context = BATContext()

    @testset "finite moments" begin
        d = TDist(3.0)

        @test BAT._mean_with_fallback(d, 1) == [mean(d)]
        @test Matrix(BAT._cov_with_fallback(d, 1)) == fill(var(d), 1, 1)

        trafo = BAT.init_adaptive_transform(
            BAT.TriangularAffineTransform(),
            batmeasure(d),
            context
        )
        @test trafo([0.0]) == [0.0]
        @test trafo([1.0]) == [sqrt(var(d))]
    end

    @testset "undefined moments" begin
        d_undefined_mean = TDist(1.0)
        @test all(isnan, BAT._mean_with_fallback(d_undefined_mean, 1))
        @test all(isnan, BAT._cov_with_fallback(d_undefined_mean, 1))
        @test_throws DomainError BAT.init_adaptive_transform(
            BAT.TriangularAffineTransform(),
            batmeasure(d_undefined_mean),
            context
        )

        d_undefined_cov = TDist(1.5)
        @test BAT._mean_with_fallback(d_undefined_cov, 1) == [mean(d_undefined_cov)]
        @test all(isinf, BAT._cov_with_fallback(d_undefined_cov, 1))
        @test_throws DomainError BAT.init_adaptive_transform(
            BAT.TriangularAffineTransform(),
            batmeasure(d_undefined_cov),
            context
        )

        d_undefined_cov_boundary = TDist(2.0)
        @test BAT._mean_with_fallback(d_undefined_cov_boundary, 1) == [mean(d_undefined_cov_boundary)]
        @test all(isinf, BAT._cov_with_fallback(d_undefined_cov_boundary, 1))
        @test_throws DomainError BAT.init_adaptive_transform(
            BAT.TriangularAffineTransform(),
            batmeasure(d_undefined_cov_boundary),
            context
        )
    end
end
