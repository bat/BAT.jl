# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using ArraysOfArrays
using Distributions
using Random
using Test

import ForwardDiff


function sample_mala_transition(x, state::MALAProposalState; seed)
    current_z = nestedview(reshape(x, length(x), 1))
    genctx = BAT.get_gencontext(BATContext(rng = Xoshiro(seed)))

    proposed_z, correction = BAT.mcmc_propose_transition(current_z, state, 1, genctx)
    return only(proposed_z), only(correction)
end

function sample_mala_transition(x, noise, grad, τ; seed)
    state = MALAProposalState(
        0.574,
        (0.5, 0.65),
        batmeasure(noise),
        z -> (0.0, grad(z)),
        τ
    )
    sample_mala_transition(x, state; seed)
end


@testset "MALAProposal" begin
    @testset "Gaussian transition density" begin
        for (x, τ, seed) in (([1.0], 1.0, 1), ([0.5, -1.0, 2.0], 0.7, 2))
            noise = MvNormal(zeros(length(x)), ones(length(x)))
            grad(z) = -z
            y, correction = sample_mala_transition(x, noise, grad, τ; seed)
            δ = y - x

            expected = (
                sum(abs2, δ - τ / 2 * grad(x)) -
                sum(abs2, -δ - τ / 2 * grad(y))
            ) / (2τ)

            @test correction ≈ expected
        end
    end

    @testset "detailed balance with configured noise" begin
        x = [0.5, -1.0]
        τ = 0.7
        noise = BAT._full_random_walk_proposal(TDist(1.0), length(x))
        logtarget(z) = -sum(abs2, z) / 2
        grad(z) = -z
        y, correction = sample_mala_transition(x, noise, grad, τ; seed = 3)
        δ = y - x

        logq_y_given_x = logpdf(noise, (δ - τ / 2 * grad(x)) / sqrt(τ)) -
            length(x) / 2 * log(τ)
        logq_x_given_y = logpdf(noise, (-δ - τ / 2 * grad(y)) / sqrt(τ)) -
            length(x) / 2 * log(τ)
        logaccept_x_to_y = min(0, logtarget(y) - logtarget(x) + correction)
        logaccept_y_to_x = min(0, logtarget(x) - logtarget(y) - correction)

        @test correction ≈ logq_x_given_y - logq_y_given_x
        @test logtarget(x) + logq_y_given_x + logaccept_x_to_y ≈
            logtarget(y) + logq_x_given_y + logaccept_y_to_x
    end

    @testset "configured proposal path" begin
        x = [0.4, -0.7]
        proposal = MALAProposal(proposaldist = TDist(2.0), τ_base = 0.7)
        target = batmeasure(MvNormal(zeros(length(x)), ones(length(x))))
        context = BATContext(ad = ForwardDiff)
        state = BAT._create_proposal_state(
            proposal, target, context, [x], identity, Xoshiro(5)
        )
        noise = convert(Distribution, state.proposaldist)
        y, correction = sample_mala_transition(x, state; seed = 5)
        τ = state.τ
        δ = y - x

        expected = logpdf(noise, (-δ + τ / 2 * y) / sqrt(τ)) -
            logpdf(noise, (δ + τ / 2 * x) / sqrt(τ))

        @test correction ≈ expected
    end

    @testset "asymmetric proposal noise" begin
        x = [0.5, -1.0]
        τ = 0.7
        noise = product_distribution(fill(Gumbel(), length(x)))
        grad(z) = -z
        y, correction = sample_mala_transition(x, noise, grad, τ; seed = 6)
        δ = y - x

        expected = logpdf(noise, (-δ - τ / 2 * grad(y)) / sqrt(τ)) -
            logpdf(noise, (δ - τ / 2 * grad(x)) / sqrt(τ))

        @test correction ≈ expected
    end

    @testset "proposal noise requires a density" begin
        samples = DensitySampleVector([[0.0], [1.0]], zeros(2), weight = ones(2))
        noise = BAT.DensitySampleMeasure(samples)
        grad(z) = -z

        @test_throws BAT.EvalException sample_mala_transition(
            [1.0], noise, grad, 1.0; seed = 4
        )
    end
end
