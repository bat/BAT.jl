# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, LinearAlgebra, Random, Random123


function _de_snooker_step_rngpart(state)
    BAT.reset_rng_counters!(state)
    purpose_width = typemax(Int16) - 2
    rng = BAT.get_rng(state.chain_state.context)
    return BAT.RNGPartition(rng, Base.OneTo(6 * purpose_width))
end


@testset "DESnookerMove" begin
    @testset "three-dimensional proposal and Hastings equation" begin
        current = [[3.0, 4.0, 0.0], zeros(3), [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
        complement_groups = ([2], [3], [4])
        candidate = fill(NaN, 3)
        proposal = BAT.DESnookerMoveProposalState(1.7, BAT.SequentialExec())
        purpose_width = typemax(Int16) - 2
        step_rngpart = BAT.RNGPartition(
            Philox4x((572, 32)), Base.OneTo(6 * purpose_width),
        )
        walkerid = 17
        oracle_rng = Philox4x((572, 321))
        BAT.set_rng!(
            oracle_rng,
            BAT._mcmc_walker_rngpart(
                step_rngpart, BAT._MCMC_COMPANION_SELECTION_PURPOSE, 1,
            ),
            walkerid,
        )
        group_orders = (
            (1, 2, 3), (1, 3, 2), (2, 1, 3),
            (2, 3, 1), (3, 1, 2), (3, 2, 1),
        )
        group_order = rand(oracle_rng, group_orders)
        reference_idx = rand(oracle_rng, complement_groups[group_order[1]])
        companion_a_idx = rand(oracle_rng, complement_groups[group_order[2]])
        companion_b_idx = rand(oracle_rng, complement_groups[group_order[3]])
        reference = current[reference_idx]
        direction = current[1] - reference
        direction_norm = norm(direction)
        unit_direction = direction / direction_norm
        displacement = proposal.scale * (
            dot(unit_direction, current[companion_a_idx]) -
            dot(unit_direction, current[companion_b_idx])
        )
        expected = current[1] + unit_direction * displacement
        expected_log_hastings = 2 * log(norm(expected - reference) / direction_norm)
        proposal_aux = BAT._propose_ensemble_candidate!!(
            candidate, proposal, current, 1, complement_groups,
            Philox4x((572, 321)), step_rngpart, 1, walkerid,
        )

        @test candidate ≈ expected
        @test BAT._ensemble_log_hastings(
            proposal, current[1], candidate, proposal_aux,
        ) ≈ expected_log_hastings

        nonfinite_candidate = fill(NaN, 3)
        @test isnothing(BAT._de_snooker_candidate!!(
            nonfinite_candidate, current[1], zeros(3),
            [floatmax(Float64), 0.0, 0.0], [-floatmax(Float64), 0.0, 0.0], 2.0,
        ))
        @test nonfinite_candidate == current[1]
    end

    @testset "degenerate directions skip target evaluation" begin
        base = batmeasure(MvNormal(zeros(1), ones(1, 1)))
        calls = Ref(0)
        target = _CountingEnsembleTarget(base, _ -> 0.0, calls)
        state = _ensemble_move_state(
            DESnookerMove(), [[0.0], [1.0], [2.0], [3.0]];
            target, rng_seed = (572, 31),
        )
        chain_state = state.chain_state
        for i in 2:4
            chain_state.current.z.v[i] .= 0.0
            chain_state.current.x.v[i] .= 0.0
        end
        current_before = deepcopy(chain_state.current)
        calls[] = 0
        p_accept = fill(NaN, 4)
        step_rngpart = _de_snooker_step_rngpart(state)
        acceptance_rngpart = BAT._mcmc_walker_rngpart(
            step_rngpart, BAT._MCMC_ACCEPTANCE_PURPOSE, 1,
        )

        BAT._evaluate_ensemble_walker!!(
            chain_state, chain_state.proposal, step_rngpart, 1, 1,
            ([2], [3], [4]), p_accept, false, acceptance_rngpart,
        )

        @test calls[] == 0
        @test p_accept[1] == 0.0
        @test !chain_state.accepted[1]
        @test chain_state.current == current_before
        @test chain_state.proposed.x[1] == chain_state.current.x[1]
        @test chain_state.proposed.z[1] == chain_state.current.z[1]
    end

    @testset "rejects a scale that underflows after conversion" begin
        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        tiny_scale = Float64(nextfloat(0.0f0)) / 2
        err = _capture_ensemble_error(() -> _ensemble_move_state(
            DESnookerMove(scale = tiny_scale), map(v -> Float32.(v), full_rank),
        ))
        @test err isa ArgumentError
    end

    @testset "four-group walker-order RNG" begin
        initial = _two_dimensional_elliptic_initial_ensemble(
            zeros(2), Matrix{Float64}(I, 2, 2), 8,
        )
        reference = BAT.mcmc_step!!(_ensemble_move_state(
            DESnookerMove(), initial; rng_seed = (572, 33),
        ))
        permutation = [5, 1, 7, 3, 8, 4, 2, 6]
        reordered = _ensemble_move_state(
            DESnookerMove(), initial[permutation]; rng_seed = (572, 33),
        )
        _set_walker_ids!(reordered, Int32.(permutation))
        reordered = BAT.mcmc_step!!(reordered)
        lhs = sortperm(getproperty.(reference.chain_state.current.x.info, :walkerid))
        rhs = sortperm(getproperty.(reordered.chain_state.current.x.info, :walkerid))

        @test reference.chain_state.current.z.v[lhs] ==
            reordered.chain_state.current.z.v[rhs]
        @test reference.chain_state.proposed.z.v[lhs] ==
            reordered.chain_state.proposed.z.v[rhs]
        @test reference.chain_state.accepted[lhs] == reordered.chain_state.accepted[rhs]
    end

    @testset "correlated affine Gaussian moments" begin
        # Limits are rounded 1.5 times 64-seed calibration maxima; 128
        # independent validation seeds had no joint failures per target
        # (95% one-sided upper bound 2.31%).
        _check_ensemble_gaussian_moments(
            DESnookerMove();
            seed = 27201,
            mean_tolerance = 0.13,
            covariance_tolerance = 0.16,
        )
    end
end
