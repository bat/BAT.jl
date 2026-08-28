# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, Statistics
using Distributions, ValueShapes, DensityInterface
using StableRNGs
import ForwardDiff
import Optim

using JLArrays, GPUArraysCore

using BAT: _lbfgs_inverse_hessians, _lbfgs_inverse_hessian, _lbfgs_curvature_update!,
    _woodbury_matrix, pathfinder_gaussian_fit,
    init_adaptive_transform, TriangularAffineTransform, PathfinderTransformInit,
    PriorApproxTransformInit
using MatrixShapedOperators: woodbury_operator, rowgram_factor

@testset "pathfinder" begin
    context = BATContext(rng = StableRNG(996770566), ad = ForwardDiff)
    rng = StableRNG(996770566)

    d = 8
    C = [0.8^abs(i-j) * sqrt((1.0+i)*(1.0+j)) for i in 1:d, j in 1:d]
    Cinv = inv(C)
    m_true = randn(rng, d)
    f_logd = x -> -dot(x - m_true, Cinv, x - m_true) / 2

    lbfgs_alg(; kwargs...) = OptimAlg(optalg = Optim.LBFGS(); kwargs...)

    @testset "gradient trace recording" begin
        r = BAT.maximize_density(f_logd, fill(4.0, d), lbfgs_alg(store_trace = true), context)
        @test isapprox(r.result, m_true, atol = 1e-6)
        xs, grads = r.trace.v, r.trace.grad_logd
        @test length(xs) == length(grads) == length(r.trace.logd)
        @test r.trace.logd ≈ f_logd.(xs)
        @test grads[end] ≈ -(Cinv * (xs[end] - m_true)) atol = 1e-6
        @test issorted(f_logd.(xs))

        # Without store_trace no trace is recorded:
        @test isnothing(BAT.maximize_density(f_logd, fill(4.0, d), lbfgs_alg(), context).trace)
    end

    @testset "inverse Hessians and factorization" begin
        # The accuracy of the inverse-Hessian estimates is limited by the
        # history length, so use a history that covers the full space:
        r = BAT.maximize_density(f_logd, fill(4.0, d), lbfgs_alg(store_trace = true), context)
        xs, grads = r.trace.v, r.trace.grad_logd
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
        # Enough ELBO draws to make the candidate selection reliable:
        fit = pathfinder_gaussian_fit(f_logd, fill(4.0, d), lbfgs_alg(), context, history_length = 20, ndraws_elbo = 30)
        @test !isnothing(fit)
        @test isapprox(fit.μ, m_true, atol = 0.05)
        @test opnorm(fit.Σ - C) / opnorm(C) < 0.05
        @test isfinite(fit.elbo)

        # With the (rank-limiting) default history length the fit is coarser
        # but must still be usable:
        fit_default = pathfinder_gaussian_fit(f_logd, fill(4.0, d), lbfgs_alg(), context)
        @test isapprox(fit_default.μ, m_true, atol = 0.1)
        @test opnorm(fit_default.Σ - C) / opnorm(C) < 0.7
    end

    @testset "candidate discipline" begin
        # A path with zero accepted L-BFGS steps carries no curvature
        # information; the initial identity estimate must never be returned
        # as a successful fit. Starting at the exact mode of a non-unit
        # Gaussian takes zero steps:
        @test isnothing(pathfinder_gaussian_fit(f_logd, copy(m_true), lbfgs_alg(), context))

        # A non-finite starting point is a path-local failure, not an error:
        x0 = fill(2.0, d)
        f_nan = x -> convert(eltype(x), NaN)
        fit_nan = @test_logs (:warn,) match_mode=:any pathfinder_gaussian_fit(f_nan, x0, lbfgs_alg(), context)
        @test isnothing(fit_nan)

        # Non-finite ELBO candidates are unusable even when optimization
        # starts from a finite point:
        for bad_logd in (Inf, -Inf, NaN)
            bad_context = BATContext(rng = StableRNG(880401), ad = ForwardDiff)
            f_bad = x -> x[1] > 1.5 ? bad_logd : -x[1]^2 / 2
            fit_bad = pathfinder_gaussian_fit(
                f_bad, [1.0], lbfgs_alg(), bad_context,
                history_length = 1, ndraws_elbo = 512
            )
            @test isnothing(fit_bad)
        end

        # Backends without a gradient trace fail at the API boundary:
        @test_throws ArgumentError pathfinder_gaussian_fit(f_logd, x0, OptimAlg(optalg = Optim.NelderMead()), context)

        # Invalid configurations fail at the API boundary:
        @test_throws ArgumentError pathfinder_gaussian_fit(f_logd, x0, lbfgs_alg(), context, history_length = 0)
        @test_throws ArgumentError pathfinder_gaussian_fit(f_logd, x0, lbfgs_alg(), context, ndraws_elbo = 0)
    end

    @testset "device array compatibility" begin
        # JLArray is a CPU-backed reference GPU array: with scalar
        # indexing disabled it fails on exactly the operations that would
        # fall back to a host implementation on a real device. It has no
        # LAPACK-level factorizations, so the ELBO scoring (which
        # factorizes) is out of reach here and only the trajectory
        # reconstruction is covered.
        nd, Jd, hind = 6, 4, 2   # hind < Jd exercises the wrapped ring buffer
        S_c, Y_c = randn(rng, nd, Jd), randn(rng, nd, Jd)
        α_c = rand(rng, nd) .+ 0.5
        s_c, y_c = randn(rng, nd), randn(rng, nd)

        H_c = _lbfgs_inverse_hessian(α_c, S_c, Y_c, hind, Jd)
        Σ_c = _woodbury_matrix(H_c.α, H_c.B, H_c.D)
        Su_c, Yu_c = copy(S_c), copy(Y_c)
        upd_c = _lbfgs_curvature_update!(Su_c, Yu_c, α_c, hind, Jd, s_c, y_c, Jd, 1e-12)

        GPUArraysCore.allowscalar(false)
        try
            α_d, S_d, Y_d = JLArray(α_c), JLArray(S_c), JLArray(Y_c)
            H_d = _lbfgs_inverse_hessian(α_d, S_d, Y_d, hind, Jd)
            @test H_d.B isa JLArray && H_d.D isa JLArray
            @test Array(H_d.α) ≈ H_c.α
            @test Array(H_d.B) ≈ H_c.B
            @test Array(H_d.D) ≈ H_c.D

            Σ_d = _woodbury_matrix(H_d.α, H_d.B, H_d.D)
            @test Σ_d isa JLArray
            @test Array(Σ_d) ≈ Σ_c

            Su_d, Yu_d = JLArray(copy(S_c)), JLArray(copy(Y_c))
            upd_d = _lbfgs_curvature_update!(Su_d, Yu_d, α_d, hind, Jd, JLArray(s_c), JLArray(y_c), Jd, 1e-12)
            @test Array(upd_d[1]) ≈ upd_c[1]
            @test upd_d[2:3] == upd_c[2:3]
            @test Array(Su_d) ≈ Su_c
            @test Array(Yu_d) ≈ Yu_c
        finally
            GPUArraysCore.allowscalar()
        end
    end

    @testset "transform initialization" begin
        # MCMC chain init partitions the context RNG, which requires a
        # counter-based RNG (unlike the StableRNG fit-test context):
        context = BATContext(rng = Philox4x((564, 45)), ad = ForwardDiff)
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
            BATContext(rng = Philox4x((564, 46)), ad = ForwardDiff),
            max_retries = 0,
        )
        @test smplres.verified
        @test smplres.n_retries == 0
    end
end
