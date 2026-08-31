# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
using Statistics
using StableRNGs

using BAT: _hmc_phasepoint, _leapfrog_step, hmc_nuts_transition

@testset "hmc_nuts" begin
    rng = StableRNG(789990641)
    f_logdgrad(q) = (-sum(abs2, q) / 2, -q)

    @testset "leapfrog trajectory" begin
        q0, p0 = randn(rng, 5), randn(rng, 5)
        z = _hmc_phasepoint(f_logdgrad, q0, p0)
        for _ in 1:25
            z = _leapfrog_step(f_logdgrad, z, 0.3)
        end
        z = _hmc_phasepoint(f_logdgrad, z.q, -z.p)
        for _ in 1:25
            z = _leapfrog_step(f_logdgrad, z, 0.3)
        end
        @test z.q ≈ q0 atol = 1e-10
        @test -z.p ≈ p0 atol = 1e-10
    end

    @testset "stationary law" begin
        q = [0.0]
        draws = Vector{Float64}(undef, 2_000)
        for i in eachindex(draws)
            z0 = _hmc_phasepoint(f_logdgrad, q, randn(rng, 1))
            q = hmc_nuts_transition(rng, f_logdgrad, z0, 0.3, 10, 1000.0).z.q
            draws[i] = q[1]
        end
        @test abs(mean(draws)) < 0.1
        @test 0.85 < std(draws, corrected = false) < 1.15
    end
end
