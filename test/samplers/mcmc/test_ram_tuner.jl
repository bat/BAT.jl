# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Statistics
using Distributions
using StableRNGs

using BAT: _rank_k_cholesky_update, _lowrankdowndate!

@testset "ram_tuner" begin
    rng = StableRNG(347912238)

    @testset "rank-k Cholesky update" begin
        # k == 1 takes the rank-one up/downdate path, k > 1 the full
        # decomposition path:
        for d in (1, 3, 25), k in (1, 4)
            Lm = Matrix(LowerTriangular(randn(rng, d, d)))
            Lm[diagind(Lm)] .= abs.(Lm[diagind(Lm)]) .+ 0.5
            L = LowerTriangular(Lm)
            u = [randn(rng, d) for _ in 1:k]
            # Mixed-sign weights, scaled like the RAM update, where
            # w_i * ||u_i||² = η * (p_accept_i - target_acceptance) > -1:
            w = [(rand(rng) - 0.5) / norm(ui)^2 for ui in u]

            L_new = _rank_k_cholesky_update(L, stack(u), w)
            M = L * (I + sum(w[i] .* u[i] .* u[i]' for i in 1:k)) * L'
            @test istril(L_new)
            @test L_new * L_new' ≈ M
            @test Matrix(L_new) ≈ Matrix(cholesky(Hermitian(M)).L)
        end
    end

    @testset "degenerate shrink stays finite" begin
        # Long streaks of very low acceptance shrink the factor towards
        # numerical singularity; the update must keep it finite instead of
        # producing non-finite values (as plain rank-one downdates would):
        u = [randn(rng, 3)]
        w = [-0.8 / norm(u[1])^2]
        L = LowerTriangular(Matrix(1.0 * I(3)))
        for _ in 1:5000
            L = _rank_k_cholesky_update(L, stack(u), w)
        end
        @test all(isfinite, Matrix(L))
        @test istril(L)
    end

    @testset "indefinite update keeps the factor" begin
        # Weights that sum up to an indefinite update, which the mean of
        # the RAM updates of the walkers can't produce:
        L = LowerTriangular(Matrix(1.0 * I(2)))
        u = [[1.0, 0.0], [1.0, 0.1]]
        w = [-0.9, -0.9]
        @test _rank_k_cholesky_update(L, stack(u), w) == L
    end

    @testset "mean of walker updates" begin
        # The mean of the RAM updates of the walkers is positive definite
        # for any acceptance probabilities, also at the maximum learning
        # rate and for the strongest shrinkage (no walker accepts):
        d, k = 8, 32
        Lm = Matrix(LowerTriangular(randn(rng, d, d)))
        Lm[diagind(Lm)] .= abs.(Lm[diagind(Lm)]) .+ 0.5
        L = LowerTriangular(Lm)
        U = randn(rng, d, k)
        for target_acceptance in (0.234, 0.9), p_accept in (rand(rng, k), zeros(k), ones(k))
            w = (p_accept .- target_acceptance) ./ vec(sum(abs2, U; dims = 1)) ./ k
            L_new = _rank_k_cholesky_update(L, U, w)
            M = L * (I + sum(w[i] .* U[:, i] .* U[:, i]' for i in 1:k)) * L'
            @test L_new * L_new' ≈ M
        end
    end

    @testset "ensemble-sized multi-walker update" begin
        # Realistic walker-ensemble sizes take the full-decomposition
        # path; with sum_i |w_i| ‖u_i‖² < 1 the exact update stays
        # positive definite, so the result must match the reference
        # decomposition:
        d, k = 16, 64
        Lm = Matrix(LowerTriangular(randn(rng, d, d)))
        Lm[diagind(Lm)] .= abs.(Lm[diagind(Lm)]) .+ 0.5
        L = LowerTriangular(Lm)
        u = [randn(rng, d) for _ in 1:k]
        w = [(rand(rng) - 0.5) / (k * norm(ui)^2) for ui in u]
        L_new = _rank_k_cholesky_update(L, stack(u), w)
        M = L * (I + sum(w[i] .* u[i] .* u[i]' for i in 1:k)) * L'
        @test istril(L_new)
        @test L_new * L_new' ≈ M
        @test Matrix(L_new) ≈ Matrix(cholesky(Hermitian(M)).L)
    end

    @testset "status-based rank-one downdate" begin
        # Matches the stdlib recurrence on valid downdates:
        Lm = Matrix(LowerTriangular(randn(rng, 6, 6)))
        Lm[diagind(Lm)] .= abs.(Lm[diagind(Lm)]) .+ 0.5
        # ‖L⁻¹v‖ = 0.5 by construction, so the downdate is always valid:
        v = 0.5 .* (Lm * normalize(randn(rng, 6)))
        A_ours = copy(Lm)
        @test _lowrankdowndate!(A_ours, copy(v))
        C_ref = Cholesky(LowerTriangular(copy(Lm)))
        lowrankdowndate!(C_ref, copy(v))
        @test LowerTriangular(A_ours) ≈ LowerTriangular(C_ref.factors)

        # Reports failure at the exact s² == 1 boundary, where a zero
        # pivot would otherwise silently produce a singular factor (as
        # the stdlib recurrence currently does):
        A = fill(1.0, 1, 1)
        @test !_lowrankdowndate!(A, [1.0])

        # Reports failure instead of throwing on indefinite downdates:
        A2 = Matrix(1.0 * I(2))
        @test !_lowrankdowndate!(A2, [1.5, 0.0])
        # NaN poisoning is also reported as failure:
        A3 = Matrix(1.0 * I(2))
        @test !_lowrankdowndate!(A3, [NaN, 0.0])
    end

    @testset "ill-conditioned near-boundary downdate" begin
        # Exercises the difference between the exact feasibility
        # condition (independent of L) and the numerical downdate on an
        # ill-conditioned factor: the result must be a valid factor of M
        # either via the downdate or via the floor - the two coincide
        # while the exact update is still positive definite:
        Lm = Matrix(LowerTriangular(ones(4, 4)))
        Lm[diagind(Lm)] .= [1.0, 1e-4, 1e-8, 1.0]
        L = LowerTriangular(Lm)
        u = [fill(0.5, 4)]
        w = [-(1 - 1e-6)]
        L_new = _rank_k_cholesky_update(L, stack(u), w)
        M = L * (I + w[1] .* u[1] .* u[1]') * L'
        @test istril(L_new)
        @test all(isfinite, Matrix(L_new))
        @test L_new * L_new' ≈ M
    end

    @testset "near-boundary downdate takes the full decomposition" begin
        # ‖L⁻¹v‖² just below one: mathematically still positive definite,
        # but inside the safety margin, so the full decomposition is used;
        # must not throw and must still give the exact result:
        L = LowerTriangular(Matrix(1.0 * I(2)))
        u = [[1.0, 0.0]]
        w = [-(1 - 1e-12)]
        L_new = _rank_k_cholesky_update(L, stack(u), w)
        @test istril(L_new)
        @test all(isfinite, Matrix(L_new))
        M = L * (I + w[1] .* u[1] .* u[1]') * L'
        @test Matrix(L_new * L_new') ≈ M
    end

    @testset "tuning step" begin
        target = batmeasure(MvNormal([1.0, -1.0, 0.5], [2.0 0.6 0.0; 0.6 1.0 0.3; 0.0 0.3 1.5]))
        context = BATContext(rng = StableRNG(789))

        # The transform after the first tuning step of a chain, and the
        # mean of the RAM updates of its walkers:
        function tuned_and_reference(nwalkers, stalled_walkers)
            algorithm = TransformedMCMC(
                proposal = RandomWalk(), pretransform = DoNotTransform(),
                transform_tuning = RAMTuning(), nwalkers = nwalkers,
            )
            v_init = BAT.bat_ensemble_initvals(target, InitFromTarget(), nwalkers, context)
            state = BAT.MCMCState(algorithm, target, 1, v_init, context)
            chain = state.chain_state
            chain.stepno = 1
            chain, proposal, step_info = BAT.mcmc_propose!!(chain, chain.proposal, BAT._mcmc_step_rng(chain))
            (; current, proposed) = chain
            for i in stalled_walkers
                proposed.z.v[i] .= current.z.v[i]
                proposed.x.v[i] .= current.x.v[i]
            end
            f = chain.f_transform
            f_tuned, _, _ = BAT.mcmc_tune_trafo_post_step!!(
                f, state.trafo_tuner_state, chain, proposal, current, proposed, step_info
            )

            L, b, p_accept = f.A, f.b, step_info.p_accept
            η = 1 # in the first tuning step
            u = proposed.z.v .- current.z.v
            w = [iszero(u[i]) ? 0.0 : η * (p_accept[i] - 0.234) / norm(u[i])^2 for i in eachindex(u)]
            M = L * (I + mean(w[i] .* u[i] .* u[i]' for i in eachindex(u))) * L'
            α = η / 10 .* p_accept
            b_ref = mean((1 .- α[i]) .* b .+ α[i] .* proposed.x.v[i] for i in eachindex(u))
            return f_tuned, M, b_ref
        end

        # A single walker follows the original RAM update:
        for nwalkers in (1, 4)
            f_tuned, M, b_ref = tuned_and_reference(nwalkers, ())
            @test f_tuned.A isa LowerTriangular
            @test f_tuned.A * f_tuned.A' ≈ M
            @test f_tuned.b ≈ b_ref
        end

        # A walker that didn't move doesn't keep the others from tuning:
        f_tuned, M, b_ref = tuned_and_reference(4, (2,))
        @test f_tuned.A * f_tuned.A' ≈ M
        @test f_tuned.b ≈ b_ref
    end

    @testset "tuning reaches the target acceptance" begin
        # A correlated Gaussian with scales over three orders of magnitude,
        # tuned from a unit transform:
        n = 20
        A = randn(rng, n, n)
        D = Diagonal(exp.(range(-1.5, 1.5, length = n)))
        target = batmeasure(MvNormal(zeros(n), Symmetric(D * (A * A' / n + 0.1 * I) * D)))

        for nwalkers in (1, 16, 256)
            algorithm = TransformedMCMC(
                proposal = RandomWalk(), pretransform = DoNotTransform(),
                adaptive_transform = BAT.TriangularAffineTransform(init = BAT.UnitTransformInit()),
                transform_tuning = RAMTuning(), nwalkers = nwalkers,
            )
            state = BAT.MCMCState(algorithm, target, 1, [zeros(n) for _ in 1:nwalkers], BATContext(rng = StableRNG(790)))
            for _ in 1:3000
                state = BAT.mcmc_step!!(state)
            end
            n_accepted, n_attempts = sum(state.chain_state.nsamples), sum(state.chain_state.nattempts)
            for _ in 1:1000
                state = BAT.mcmc_step!!(state)
            end
            n_accepted = sum(state.chain_state.nsamples) - n_accepted
            n_attempts = sum(state.chain_state.nattempts) - n_attempts
            @test 0.15 < n_accepted / n_attempts < 0.35
        end
    end

    @testset "requires a lower-triangular factor" begin
        A = randn(rng, 5, 5) + 3 * I
        u = [randn(rng, 5)]
        w = [0.2 / norm(u[1])^2]
        @test_throws MethodError _rank_k_cholesky_update(A, stack(u), w)
    end
end
