# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Statistics
using Distributions, DensityInterface
using Random123
import ForwardDiff

using BAT: MALAProposal, _mala_innovation_dist, _mala_log_proposal_ratio,
    batmeasure

@testset "mala" begin
    @testset "proposal log ratio" begin
        τ = 0.3
        Δ = [0.4, -0.7]
        g_x = [0.2, -0.5]
        g_y = [-0.3, 0.6]
        pm = batmeasure(_mala_innovation_dist(Normal(), length(Δ)))
        h = only(_mala_log_proposal_ratio(pm, τ, [Δ], [g_x], [g_y]))
        h_ref = (sum(abs2, Δ .- τ / 2 .* g_x) -
            sum(abs2, .-Δ .- τ / 2 .* g_y)) / (2τ)
        @test h ≈ h_ref
    end

    @testset "stationary law" begin
        μ = [0.4, -0.7]
        Σ = [1.0 0.35; 0.35 1.5]
        target = batmeasure(MvNormal(μ, Σ))
        algorithm = TransformedMCMC(
            proposal = MALAProposal(τ_base = 0.7),
            proposal_tuning = BAT.NoMCMCProposalTuning(),
            adaptive_transform = BAT.NoAdaptiveTransform(),
            pretransform = DoNotTransform(),
            nchains = 1,
            nwalkers = 1,
        )
        state = BAT.MCMCState(
            algorithm,
            target,
            1,
            [zeros(2)],
            BATContext(rng = Philox4x((564, 1305)), ad = ForwardDiff),
        )
        draws = Matrix{Float64}(undef, 2, 3_500)
        for i in 1:4_000
            state = BAT.mcmc_step!!(state)
            i > 500 && (draws[:, i - 500] = only(state.chain_state.current.x.v))
        end
        @test vec(mean(draws; dims = 2)) ≈ μ atol = 0.04
        @test cov(draws; dims = 2) ≈ Σ atol = 0.07
    end

    @testset "transformed gradient" begin
        f = BAT.MulAdd(Diagonal([2.0, 0.5]), [1.0, -2.0])
        target = batmeasure(MvNormal(zeros(2), I))
        algorithm = TransformedMCMC(
            proposal = MALAProposal(τ_base = 0.2),
            proposal_tuning = BAT.NoMCMCProposalTuning(),
            adaptive_transform = BAT.CustomTransform(f),
            pretransform = DoNotTransform(),
            nchains = 1,
            nwalkers = 1,
        )
        state = BAT.MCMCState(
            algorithm,
            target,
            1,
            [[0.1, -0.2]],
            BATContext(rng = Philox4x((564, 1306)), ad = ForwardDiff),
        )
        state = BAT.mcmc_step!!(state)
        proposal = BAT.get_active_proposal(state.chain_state.proposal)
        gradient = only(proposal.grad_cache.grads_curr)
        x = only(state.chain_state.current.x.v)
        z = only(state.chain_state.current.z.v)
        @test x == f(z)
        @test only(state.chain_state.current.x.logd) == logdensityof(target, x)
        @test gradient == f.A' * (-x)
    end
end
