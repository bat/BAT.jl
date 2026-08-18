# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, Statistics
using Distributions, ValueShapes
using StableRNGs
import ForwardDiff

using BAT: DenseFisherEstimator, DiagonalFisherEstimator, DriftCommitSchedule,
    FisherTransformTuning, _new_moments, _moments_update!, _fisher_geometry,
    _spd_riccati_solve, _transform_drift

@testset "fisher_tuner" begin
    rng = StableRNG(438621057)

    @testset "Fisher geometry recovery" begin
        # For a Gaussian N(μ, Σ) the score is α = -Σ⁻¹(x - μ), and the
        # Fisher-optimal affine geometry is exactly G = Σ, μ* = μ:
        d = 4
        A_true = LowerTriangular(Matrix(I(d)) + 0.4 * randn(rng, d, d))
        Σ = Matrix(Symmetric(A_true * A_true' + 0.1 * I))
        Σinv = inv(Σ)
        μ_true = randn(rng, d)

        acc_dense = _new_moments(DenseFisherEstimator(), d)
        acc_diag = _new_moments(DiagonalFisherEstimator(), d)
        for _ in 1:10^4
            x = μ_true .+ cholesky(Σ).L * randn(rng, d)
            α = -Σinv * (x .- μ_true)
            _moments_update!(acc_dense, x, α)
            _moments_update!(acc_diag, x, α)
        end

        G, μ = _fisher_geometry(DenseFisherEstimator(), acc_dense, 1e-5)
        @test opnorm(Matrix(G) - Σ) / opnorm(Σ) < 0.1
        @test isapprox(μ, μ_true, atol = 0.2)

        # The diagonal variant recovers sqrt(Var(x) / Var(α)) per dimension:
        G_diag, _ = _fisher_geometry(DiagonalFisherEstimator(), acc_diag, 1e-5)
        var_x = acc_diag.M2_x ./ (acc_diag.n - 1)
        var_g = acc_diag.M2_g ./ (acc_diag.n - 1)
        @test diag(G_diag) ≈ sqrt.((var_x .+ 1e-5) ./ (var_g .+ 1e-5))
    end

    @testset "Riccati solve and drift metric" begin
        d = 5
        R1, R2 = randn(rng, d, d), randn(rng, d, d)
        C_x = Symmetric(R1 * R1' + 0.1 * I)
        C_g = Symmetric(R2 * R2' + 0.1 * I)
        G = _spd_riccati_solve(C_x, C_g)
        @test Matrix(G * C_g * G) ≈ Matrix(C_x)

        A = LowerTriangular(Matrix(cholesky(Symmetric(Matrix(G))).L))
        # The installed geometry itself has zero drift:
        @test _transform_drift(A, G) < 1e-8
        # A pure rescaling G -> c² G has drift |log(c²)| √d:
        c2 = 4.0
        @test _transform_drift(A, Symmetric(c2 * Matrix(G))) ≈ log(c2) * sqrt(d)
    end

    @testset "guards" begin
        context = BATContext(ad = ForwardDiff)
        target = unshaped(batmeasure(NamedTupleDist(a = Normal(), b = Normal())))
        # Fisher tuning requires a gradient-based proposal:
        alg_rw = TransformedMCMC(
            proposal = RandomWalk(), transform_tuning = FisherTransformTuning(),
            nchains = 1, nsteps = 100
        )
        @test_throws ArgumentError bat_sample(target, alg_rw, context)
    end

    @testset "end-to-end geometry learning" begin
        context = BATContext(ad = ForwardDiff)
        Σ = [4.0 1.2 0.0; 1.2 2.0 -0.5; 0.0 -0.5 1.0]
        objective = MvNormal([1.0, -2.0, 0.5], Σ)
        target = batmeasure(objective)

        # No pretransform: the Fisher tuner has to learn the full geometry:
        alg = TransformedMCMC(
            proposal = HamiltonianMC(),
            pretransform = DoNotTransform(),
            nchains = 2,
            nsteps = 10^4
        )
        @test alg.transform_tuning isa FisherTransformTuning

        em = evalmeasure(target, alg, context)
        smpls = BAT.samplesof(em)
        @test BAT.test_dist_samples(objective, smpls, context)

        # The learned affine transform reproduces the target geometry:
        gen = BAT.samplegenof(em)
        f = gen.chain_states[1].f_transform
        G_learned = Matrix(f.A * f.A')
        @test opnorm(G_learned - Σ) / opnorm(Σ) < 0.35
        @test isapprox(f.b, mean(objective), atol = 0.5)

        # Trajectory diagnostics are recorded in the evaluation info:
        diags = BAT.evalinfo(em).result.chain_diagnostics
        @test length(diags) == 2
        @test all(d -> d.n_transitions > 0, diags)
        @test all(d -> 0 < d.mean_p_accept <= 1, diags)
        @test all(d -> d.n_leapfrog > 0, diags)
    end
end
