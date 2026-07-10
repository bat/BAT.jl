# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, Statistics
using Distributions, ValueShapes, DensityInterface
using StableRNGs
import ForwardDiff

using BAT: _lbfgs_trace, _lbfgs_inverse_hessians, pathfinder_gaussian_fit,
    init_adaptive_transform, TriangularAffineTransform, PathfinderTransformInit,
    PriorApproxTransformInit
using MatrixShapedOperators: woodbury_operator, rowgram_factor

@testset "pathfinder" begin
    rng = StableRNG(996770566)

    d = 8
    C = [0.8^abs(i-j) * sqrt((1.0+i)*(1.0+j)) for i in 1:d, j in 1:d]
    Cinv = inv(C)
    m_true = randn(rng, d)
    f_logd = x -> -dot(x - m_true, Cinv, x - m_true) / 2
    f_logdgrad = x -> (f_logd(x), -(Cinv * (x - m_true)))

    @testset "L-BFGS trace" begin
        xs, grads = _lbfgs_trace(f_logdgrad, fill(4.0, d), maxiters = 200, history_length = 6)
        @test length(xs) == length(grads)
        @test issorted(f_logd.(xs))
        @test isapprox(last(xs), m_true, atol = 1e-6)
    end

    @testset "inverse Hessians and factorization" begin
        # The accuracy of the inverse-Hessian estimates is limited by the
        # history length, so use a history that covers the full space:
        xs, grads = _lbfgs_trace(f_logdgrad, fill(4.0, d), maxiters = 200, history_length = 20)
        Hs = _lbfgs_inverse_hessians(xs, grads, history_length = 20)
        @test length(Hs) == length(xs)

        # For a Gaussian target the final inverse-Hessian estimate
        # approximates the covariance:
        (; α, B, D) = last(Hs)
        Σ = B * D * B' + Diagonal(α)
        @test opnorm(Σ - C) / opnorm(C) < 0.05

        # The Woodbury gram factor satisfies L * Lᵀ == Σ:
        F = rowgram_factor(woodbury_operator(Diagonal(α), B, Symmetric(D)))
        L = Matrix(F)
        @test L * L' ≈ Σ
        @test first(logabsdet(F)) ≈ logdet(Symmetric(Σ)) / 2
    end

    @testset "gaussian fit" begin
        fit = pathfinder_gaussian_fit(rng, f_logd, f_logdgrad, fill(4.0, d), history_length = 20)
        @test !isnothing(fit)
        @test isapprox(fit.μ, m_true, atol = 0.05)
        @test opnorm(fit.Σ - C) / opnorm(C) < 0.05
        @test isfinite(fit.elbo)

        # With the (rank-limiting) default history length the fit is coarser
        # but must still be usable:
        fit_default = pathfinder_gaussian_fit(rng, f_logd, f_logdgrad, fill(4.0, d))
        @test isapprox(fit_default.μ, m_true, atol = 0.1)
        @test opnorm(fit_default.Σ - C) / opnorm(C) < 0.7
    end

    @testset "transform initialization" begin
        context = BATContext(ad = ForwardDiff)
        objective = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))
        target = unshaped(batmeasure(objective))
        Σ_true = cov(unshaped(objective))

        v_init = [randn(rng, 3) .* 2 for _ in 1:4]
        at = TriangularAffineTransform(init = PathfinderTransformInit())
        f = init_adaptive_transform(at, target, v_init, context)
        @test istril(f.A)
        @test isapprox(f.b, mean(unshaped(objective)), atol = 0.5)
        @test opnorm(f.A * f.A' - Σ_true) / opnorm(Σ_true) < 0.5

        # Requires initial positions and an AD backend:
        @test_throws ArgumentError init_adaptive_transform(at, target, context)
        @test_throws Exception init_adaptive_transform(at, target, v_init, BATContext())

        # Default initialization is unchanged:
        @test TriangularAffineTransform().init isa PriorApproxTransformInit

        smplres = BAT.sample_and_verify(
            batmeasure(objective),
            TransformedMCMC(
                proposal = RandomWalk(),
                adaptive_transform = at,
                pretransform = DoNotTransform(),
                nsteps = 10^4
            ),
            objective,
            context
        )
        @test smplres.verified
    end
end
