# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using StableRNGs

using BAT: _rank_k_cholesky_update, _lowrankdowndate!

@testset "ram_tuner" begin
    rng = StableRNG(347912238)

    @testset "rank-k Cholesky update" begin
        # k == 1 takes the rank-one up/downdate path, k > 1 the full
        # modified-decomposition path:
        for d in (1, 3, 25), k in (1, 4)
            Lm = Matrix(LowerTriangular(randn(rng, d, d)))
            Lm[diagind(Lm)] .= abs.(Lm[diagind(Lm)]) .+ 0.5
            L = LowerTriangular(Lm)
            u = [randn(rng, d) for _ in 1:k]
            # Mixed-sign weights, scaled like the RAM update, where
            # w_i * ||u_i||² = η * (p_accept_i - target_acceptance) > -1:
            w = [(rand(rng) - 0.5) / norm(ui)^2 for ui in u]

            L_new = _rank_k_cholesky_update(L, u, w)
            M = L * (I + sum(w[i] .* u[i] .* u[i]' for i in 1:k)) * L'
            @test istril(L_new)
            @test L_new * L_new' ≈ M
            @test Matrix(L_new) ≈ Matrix(cholesky(Hermitian(M)).L)
        end
    end

    @testset "degenerate shrink stays finite" begin
        # Long streaks of very low acceptance shrink the factor towards
        # numerical singularity; the update must floor it instead of
        # producing non-finite values (as plain rank-one downdates would):
        u = [randn(rng, 3)]
        w = [-0.8 / norm(u[1])^2]
        L = LowerTriangular(Matrix(1.0 * I(3)))
        for _ in 1:5000
            L = _rank_k_cholesky_update(L, u, w)
        end
        @test all(isfinite, Matrix(L))
        @test istril(L)
    end

    @testset "indefinite multi-walker update floors" begin
        # Aggregated mixed-sign walker updates can make the exact update
        # indefinite, the modified decomposition must floor it:
        L = LowerTriangular(Matrix(1.0 * I(2)))
        u = [[1.0, 0.0], [1.0, 0.1]]
        w = [-0.9, -0.9]
        L_new = _rank_k_cholesky_update(L, u, w)
        @test istril(L_new)
        @test all(isfinite, Matrix(L_new))
        @test isposdef(Matrix(L_new * L_new'))
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
        L_new = _rank_k_cholesky_update(L, u, w)
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
        L_new = _rank_k_cholesky_update(L, u, w)
        M = L * (I + w[1] .* u[1] .* u[1]') * L'
        @test istril(L_new)
        @test all(isfinite, Matrix(L_new))
        @test L_new * L_new' ≈ M
    end

    @testset "near-boundary downdate takes the floor path" begin
        # ‖L⁻¹v‖² just below one: mathematically still positive definite,
        # but inside the safety margin, so the modified decomposition is
        # used; must not throw and must still give the exact result:
        L = LowerTriangular(Matrix(1.0 * I(2)))
        u = [[1.0, 0.0]]
        w = [-(1 - 1e-12)]
        L_new = _rank_k_cholesky_update(L, u, w)
        @test istril(L_new)
        @test all(isfinite, Matrix(L_new))
        M = L * (I + w[1] .* u[1] .* u[1]') * L'
        @test Matrix(L_new * L_new') ≈ M
    end

    @testset "requires a lower-triangular factor" begin
        A = randn(rng, 5, 5) + 3 * I
        u = [randn(rng, 5)]
        w = [0.2 / norm(u[1])^2]
        @test_throws MethodError _rank_k_cholesky_update(A, u, w)
    end
end
