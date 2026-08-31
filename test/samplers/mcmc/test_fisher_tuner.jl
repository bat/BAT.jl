# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random
using StableRNGs

using BAT: DenseFisherEstimator, LowRankFisherEstimator,
    _new_moments, _moments_update!, _fisher_geometry

@testset "fisher_tuner" begin
    rng = StableRNG(438621057)

    @testset "Gaussian geometry recovery" begin
        d = 4
        A = LowerTriangular(Matrix(I(d)) + 0.4 * randn(rng, d, d))
        covariance = Matrix(Symmetric(A * A' + 0.1 * I))
        covariance_inv = inv(covariance)
        mean_true = randn(rng, d)
        moments = _new_moments(DenseFisherEstimator(), d)
        factor = cholesky(covariance).L

        for _ in 1:2_000
            x = mean_true .+ factor * randn(rng, d)
            score = -covariance_inv * (x - mean_true)
            _moments_update!(moments, x, score)
        end

        geometry, mean = _fisher_geometry(DenseFisherEstimator(), moments, 1e-5)
        @test opnorm(Matrix(geometry) - covariance) / opnorm(covariance) < 0.2
        @test mean ≈ mean_true atol = 0.35
    end

    @testset "low-rank geometry recovery" begin
        d = 4
        direction = normalize(ones(d))
        covariance = Matrix(Symmetric(Diagonal([1.0, 2.0, 0.5, 1.5]) + 8.0 * direction * direction'))
        covariance_inv = inv(covariance)
        estimator = LowRankFisherEstimator(1.5, 1)
        moments = _new_moments(estimator, d)
        samples = cholesky(Symmetric(covariance)).L * hcat(Matrix(I, d, d), -Matrix(I, d, d))
        scores = -covariance_inv * samples

        for i in axes(samples, 2)
            _moments_update!(moments, samples[:, i], scores[:, i])
        end

        diagonal, _ = BAT._fisher_diagonal_geometry(moments, 1e-5)
        candidate = BAT._fit_lowrank_candidate(estimator, diag(diagonal), samples, scores, 1e-5)
        geometry = BAT._lowrank_geometry(diag(diagonal), candidate)
        @test opnorm(Matrix(geometry) - covariance) / opnorm(covariance) < 0.3
    end
end
