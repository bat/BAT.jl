# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Distributions
using Test
using ValueShapes

function _adaptive_multi_prop_state(picking_rule)
    context = BATContext(rng = BAT.Random123.Philox4x((0, 1)))
    target = unshaped(batmeasure(Normal()))
    n_proposals = length(
        picking_rule isa Categorical ? picking_rule.p : picking_rule,
    )
    proposal = MCMCMultiProposal(
        [MetropolisHastings() for _ in 1:n_proposals],
        picking_rule,
    )
    algorithm = TransformedMCMC(
        proposal = proposal,
        proposal_tuning = AdaptiveMultiPropTuning(picking_socket = 0.8),
        pretransform = DoNotTransform(),
        adaptive_transform = BAT.NoAdaptiveTransform(),
        convergence = AssumeConvergence(),
        nchains = 1,
        nwalkers = 1,
        nsteps = 1,
    )
    v_init = BAT.bat_ensemble_initvals(target, InitFromTarget(), 1, context)
    return BAT.MCMCState(algorithm, target, 1, unshaped.(v_init), context)
end

_picking_probabilities(picking_rule::Vector) = picking_rule ./ sum(picking_rule)
_picking_probabilities(picking_rule::Categorical) = picking_rule.p

@testset "adaptive multi-proposal tuner" begin
    @testset "post-step and post-cycle integration" begin
        for picking_rule in ([2, 1], Categorical([2 / 3, 1 / 3]))
            caller_probabilities = copy(
                picking_rule isa Categorical ? picking_rule.p : picking_rule,
            )
            state = _adaptive_multi_prop_state(picking_rule)
            output = BAT._empty_chain_outputs(state)

            state = BAT.mcmc_iterate!!(
                output,
                state;
                max_nsteps = 2,
                nonzero_weights = false,
            )
            post_step_rule = state.chain_state.proposal.picking_rule
            @test typeof(post_step_rule) == typeof(picking_rule)
            @test sum(_picking_probabilities(post_step_rule)) ≈ 1

            state = BAT.mcmc_tune_post_cycle!!(state, output)
            post_cycle_rule = state.chain_state.proposal.picking_rule
            @test typeof(post_cycle_rule) == typeof(picking_rule)
            @test sum(_picking_probabilities(post_cycle_rule)) ≈ 1
            @test all(isfinite, _picking_probabilities(post_cycle_rule))

            caller_storage = picking_rule isa Categorical ? picking_rule.p : picking_rule
            @test caller_storage == caller_probabilities
        end
    end

    @testset "integer-vector rules preserve their total weight" begin
        picking_rule = [2, 1]

        post_step_rule = BAT._tune_picking_rule(picking_rule, 1.0, 2, 0.0, 2)
        @test post_step_rule == [1, 2]
        @test picking_rule == [2, 1]

        disabled_rule = BAT._qualify_picking_rule(picking_rule, [1.0, 0.0], 0.0, 2)
        @test disabled_rule == [3, 0]
        @test sum(disabled_rule) == sum(picking_rule)

        reenabled_rule = BAT._qualify_picking_rule(disabled_rule, [1.0, 1.0], 0.8, 2)
        @test all(>(0), reenabled_rule)
        @test sum(reenabled_rule) == sum(picking_rule)

        zero_quality_rule = BAT._qualify_picking_rule(
            picking_rule,
            [0.0, 0.0],
            0.8,
            2,
        )
        @test all(>(0), zero_quality_rule)
        @test sum(zero_quality_rule) == sum(picking_rule)
        @test picking_rule == [2, 1]
    end

    @testset "integer-vector finalization preserves total weight" begin
        state = _adaptive_multi_prop_state([1, 1, 1])
        state.chain_state.stepno = 10
        state.chain_state.nsamples .= [2, 0, 2]

        state = BAT.mcmc_tuning_finalize!!(state)
        finalized_rule = state.chain_state.proposal.picking_rule
        @test finalized_rule[2] == 0
        @test all(>(0), finalized_rule[[1, 3]])
        @test sum(finalized_rule) == 3
    end

    @testset "Categorical rules do not alias caller probability storage" begin
        caller_probabilities = [2 / 3, 1 / 3]
        picking_rule = Categorical(caller_probabilities)

        post_step_rule = BAT._tune_picking_rule(picking_rule, 1.0, 2, 0.0, 2)
        @test post_step_rule.p ≈ [0.4, 0.6]
        @test picking_rule.p == [2 / 3, 1 / 3]
        @test caller_probabilities == [2 / 3, 1 / 3]

        disabled_rule = BAT._qualify_picking_rule(picking_rule, [1.0, 0.0], 0.0, 2)
        @test disabled_rule.p ≈ [1.0, 0.0]

        reenabled_rule = BAT._qualify_picking_rule(disabled_rule, [1.0, 1.0], 0.8, 2)
        @test reenabled_rule.p ≈ [0.6, 0.4]
        @test sum(reenabled_rule.p) ≈ 1

        zero_quality_rule = BAT._qualify_picking_rule(
            picking_rule,
            [0.0, 0.0],
            0.8,
            2,
        )
        @test zero_quality_rule.p ≈ [0.5, 0.5]
        @test picking_rule.p == [2 / 3, 1 / 3]
        @test caller_probabilities == [2 / 3, 1 / 3]
    end
end
