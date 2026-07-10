# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra
using StableRNGs

using BAT: _rank_k_cholesky_update

@testset "ram_tuner" begin
    rng = StableRNG(347912238)

    @testset "rank-k Cholesky update" begin
        for d in (1, 3, 25), k in (1, 4)
            L = Matrix(LowerTriangular(randn(rng, d, d)))
            L[diagind(L)] .= abs.(L[diagind(L)]) .+ 0.5
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

    @testset "non-triangular fallback" begin
        A = randn(rng, 5, 5) + 3 * I
        u = [randn(rng, 5)]
        w = [0.2 / norm(u[1])^2]
        L_new = _rank_k_cholesky_update(A, u, w)
        M = A * (I + w[1] .* u[1] .* u[1]') * A'
        @test istril(L_new)
        @test L_new * L_new' ≈ M
    end
end
