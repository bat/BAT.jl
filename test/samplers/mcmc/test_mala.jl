# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, Statistics
using Distributions, ValueShapes, DensityInterface
using StableRNGs
import ForwardDiff

using BAT: MALAProposal, StepSizeAdaptor, _mala_innovation_dist, _mala_log_proposal_ratio, batmeasure

@testset "mala" begin
    rng = StableRNG(902114857)

    @testset "innovation distribution" begin
        # The Langevin innovation acts at unit scale, unlike random-walk
        # proposal distributions (which carry the optimal-scaling factor):
        d = _mala_innovation_dist(Normal(), 5)
        @test length(d) == 5
        @test var(d) == fill(1.0, 5)
    end

    @testset "exact proposal log ratio" begin
        τ = 0.3
        n = 3
        Δ = randn(rng, n)
        g_x = randn(rng, n)
        g_y = randn(rng, n)

        # Gaussian innovation: must match the closed-form Gaussian MALA
        # Hastings correction with the τ/2 drift:
        pm = batmeasure(_mala_innovation_dist(Normal(), n))
        h = only(_mala_log_proposal_ratio(pm, τ, [Δ], [g_x], [g_y]))
        h_ref = (sum(abs2, Δ .- τ/2 .* g_x) - sum(abs2, .-Δ .- τ/2 .* g_y)) / (2τ)
        @test h ≈ h_ref

        # Non-Gaussian innovation: must match the exact log-density ratio
        # of the actual innovation distribution:
        pm_t = batmeasure(_mala_innovation_dist(TDist(4.0), n))
        ξ_fwd = (Δ .- τ/2 .* g_x) ./ sqrt(τ)
        ξ_rev = (.-Δ .- τ/2 .* g_y) ./ sqrt(τ)
        h_t = only(_mala_log_proposal_ratio(pm_t, τ, [Δ], [g_x], [g_y]))
        @test h_t ≈ logdensityof(pm_t, ξ_rev) - logdensityof(pm_t, ξ_fwd)
    end

    @testset "sampling correctness" begin
        context = BATContext(ad = ForwardDiff)
        Σ = [1.0 0.6; 0.6 2.0]
        objective = MvNormal([1.0, -1.0], Σ)

        # Default MALA (Gaussian innovation, Fisher transform tuning):
        smplres = BAT.sample_and_verify(
            batmeasure(objective),
            TransformedMCMC(proposal = MALAProposal(), pretransform = DoNotTransform(), nsteps = 3 * 10^4),
            objective,
            context
        )
        @test smplres.verified

        # The Fisher tuner sees coherent position/score pairs also under
        # MALA rejections, so the learned geometry matches the target:
        cs = BAT.samplegenof(smplres.evaluated).chain_states[1]
        f = cs.f_transform
        G_learned = Matrix(f.A * f.A')
        @test opnorm(G_learned - Σ) / opnorm(Σ) < 0.5

        # The step-scale adaptor steers the acceptance rate into the
        # target region (for MALA the state-movement rate is the
        # acceptance rate):
        @test 0.4 < BAT.eff_acceptance_ratio(cs) < 0.75

        # A heavy-tailed innovation is a valid generalized Langevin-MH
        # proposal now that the exact proposal densities are used:
        smplres_t = BAT.sample_and_verify(
            batmeasure(objective),
            TransformedMCMC(proposal = MALAProposal(proposaldist = TDist(4.0)), pretransform = DoNotTransform(), nsteps = 3 * 10^4),
            objective,
            context
        )
        @test smplres_t.verified
    end

    @testset "step scale adaptation" begin
        # Dual averaging moves τ against a persistent acceptance
        # imbalance:
        tuner_lo = BAT.MALAStepSizeTunerState(StepSizeAdaptor(), 0, log(10 * 0.5), 0.0, 0.0)
        τ = 0.5
        for _ in 1:200
            τ = BAT._dual_averaging_step!(tuner_lo, 0.574, 0.1)
        end
        @test τ < 0.5

        tuner_hi = BAT.MALAStepSizeTunerState(StepSizeAdaptor(), 0, log(10 * 0.5), 0.0, 0.0)
        τ = 0.5
        for _ in 1:200
            τ = BAT._dual_averaging_step!(tuner_hi, 0.574, 0.95)
        end
        @test τ > 0.5
    end
end
