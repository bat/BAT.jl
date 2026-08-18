# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, LinearAlgebra, Statistics
using StableRNGs

using BAT: HMCPhasePoint, _hmc_phasepoint, _leapfrog_step, _logaddexp,
    hmc_nuts_transition, hmc_find_good_stepsize, _stan_adaptation_windows

@testset "hmc_nuts" begin
    rng = StableRNG(789990641)

    # Standard normal target with hand-coded gradient:
    f_logdgrad(q) = (-sum(abs2, q) / 2, -q)

    @testset "phase points" begin
        q, p = [1.0, 2.0], [0.5, -0.5]
        z = @inferred(_hmc_phasepoint(f_logdgrad, q, p))
        @test z isa HMCPhasePoint
        @test z.logd == -2.5
        @test z.grad == -q
        @test z.H ≈ 0.25 - z.logd

        z_bad = _hmc_phasepoint(q, p, NaN, [1.0, 1.0])
        @test z_bad.logd == -Inf && z_bad.H == Inf
        z_badgrad = _hmc_phasepoint(q, p, 1.0, [NaN, 1.0])
        @test z_badgrad.H == Inf
    end

    @testset "leapfrog" begin
        z0 = _hmc_phasepoint(f_logdgrad, randn(rng, 5), randn(rng, 5))

        # Energy conservation for small steps:
        z = z0
        for _ in 1:100
            z = _leapfrog_step(f_logdgrad, z, 0.01)
        end
        @test abs(z.H - z0.H) < 1e-4

        # Reversibility:
        z_fwd = _leapfrog_step(f_logdgrad, z0, 0.1)
        z_back = _leapfrog_step(f_logdgrad, z_fwd, -0.1)
        @test z_back.q ≈ z0.q atol = 1e-12
        @test z_back.p ≈ z0.p atol = 1e-12
    end

    @testset "logaddexp" begin
        @test _logaddexp(log(2.0), log(3.0)) ≈ log(5.0)
        @test _logaddexp(-Inf, 0.0) == 0.0
        @test _logaddexp(-Inf, -Inf) == -Inf
    end

    @testset "find_good_stepsize" begin
        stepsize = hmc_find_good_stepsize(rng, f_logdgrad, randn(rng, 3))
        z0 = _hmc_phasepoint(f_logdgrad, randn(rng, 3), randn(rng, 3))
        # For a standard normal, reasonable step sizes are O(1):
        @test 0.05 < stepsize < 5
    end

    @testset "nuts_transition" begin
        n_dims = 3
        stepsize = 0.3

        # Statistical correctness on a standard normal:
        n_samples = 10^4
        q = zeros(n_dims)
        Q = Vector{Vector{Float64}}()
        n_div = 0
        for _ in 1:n_samples
            z0 = _hmc_phasepoint(f_logdgrad, q, randn(rng, n_dims))
            t = hmc_nuts_transition(rng, f_logdgrad, z0, stepsize, 10, 1000.0)
            @test 0 <= t.p_accept <= 1
            @test t.n_leapfrog >= 1
            q = t.z.q
            n_div += t.divergent
            push!(Q, q)
        end
        @test n_div == 0
        X = stack(Q)
        @test isapprox(vec(mean(X, dims = 2)), zeros(n_dims), atol = 0.1)
        @test isapprox(cov(X, dims = 2), I(n_dims), atol = 0.15, norm = M -> maximum(abs, M))

        # Divergence detection with an unstable step size:
        z0 = _hmc_phasepoint(f_logdgrad, fill(10.0, n_dims), randn(rng, n_dims))
        t_div = hmc_nuts_transition(rng, f_logdgrad, z0, 1e3, 10, 1000.0)
        @test t_div.divergent
        @test t_div.z.q == z0.q

        # max_depth caps the trajectory length:
        z0 = _hmc_phasepoint(f_logdgrad, randn(rng, n_dims), randn(rng, n_dims))
        t_shallow = hmc_nuts_transition(rng, f_logdgrad, z0, 0.01, 3, 1000.0)
        @test t_shallow.depth <= 3
        @test t_shallow.n_leapfrog <= 2^3 + 2^3 - 1
    end

    @testset "stan_adaptation_windows" begin
        # Reference values match Stan's windowed_adaptation (and AdvancedHMC):
        @test _stan_adaptation_windows(75, 50, 25, 1000) == (76, 950, [100, 150, 250, 450, 950])
        @test _stan_adaptation_windows(75, 50, 25, 150) == (76, 100, [100])
        # No complete window fits:
        @test _stan_adaptation_windows(75, 50, 25, 100) == (76, 50, Int[])
    end
    @testset "leapfrog reversibility" begin
        # Leapfrog is time-reversible: integrating forward, flipping the
        # momentum, integrating the same number of steps and flipping again
        # returns to the initial phase point up to floating-point error:
        q0, p0 = randn(rng, 5), randn(rng, 5)
        z = _hmc_phasepoint(f_logdgrad, q0, p0)
        ε = 0.3
        n = 25
        for _ in 1:n
            z = _leapfrog_step(f_logdgrad, z, ε)
        end
        z = _hmc_phasepoint(f_logdgrad, z.q, -z.p)
        for _ in 1:n
            z = _leapfrog_step(f_logdgrad, z, ε)
        end
        @test isapprox(z.q, q0, atol = 1e-10)
        @test isapprox(-z.p, p0, atol = 1e-10)
    end

    @testset "leapfrog energy-error scaling" begin
        # For fixed integration time the global energy error of leapfrog
        # scales like ε², so halving the step size (doubling the step
        # count) must shrink it by roughly a factor of four:
        q0, p0 = randn(rng, 5), randn(rng, 5)
        T_total = 2.0
        function energy_error(nsteps)
            z = _hmc_phasepoint(f_logdgrad, copy(q0), copy(p0))
            ε = T_total / nsteps
            H0 = z.H
            for _ in 1:nsteps
                z = _leapfrog_step(f_logdgrad, z, ε)
            end
            return abs(z.H - H0)
        end
        e1, e2, e3 = energy_error(50), energy_error(100), energy_error(200)
        @test e2 < e1 / 2
        @test e3 < e2 / 2
    end

    @testset "affine equivalence" begin
        # Identity-metric NUTS on the pullback of N(0, Σ) through
        # A = chol(Σ) is the exact same Markov process as NUTS on N(0, I):
        # given the same RNG stream, the transitions must be identical:
        d = 4
        R = randn(rng, d, d)
        Σ = Matrix(Symmetric(R * R' + 0.5 * I))
        A = cholesky(Σ).L
        Σinv = inv(Σ)
        # Pullback of N(0, Σ) through x = A z (constant LADJ omitted, it
        # does not affect transitions):
        fg_pullback(z) = begin
            x = A * z
            g_x = -(Σinv * x)
            (-dot(x, Σinv, x) / 2, A' * g_x)
        end
        z0 = randn(rng, d)
        for eps in (0.2, 0.7)
            rng1, rng2 = StableRNG(20260821), StableRNG(20260821)
            t1 = hmc_nuts_transition(rng1, fg_pullback, _hmc_phasepoint(fg_pullback, copy(z0), randn(StableRNG(11), d)), eps, 10, 1000.0)
            t2 = hmc_nuts_transition(rng2, f_logdgrad, _hmc_phasepoint(f_logdgrad, copy(z0), randn(StableRNG(11), d)), eps, 10, 1000.0)
            @test t1.z.q ≈ t2.z.q
            @test t1.p_accept ≈ t2.p_accept
            @test t1.depth == t2.depth
            @test t1.n_leapfrog == t2.n_leapfrog
        end
    end

end
