# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, Distributions, Random, Random123, ValueShapes


@testset "DEMove" begin
    @testset "seeded noisy proposal equation" begin
        current = [[1.0, -2.0], [3.0, 4.0], [-1.0, 5.0], [6.0, -3.0]]
        proposal = BAT.DEMoveProposalState(0.75, 0.2, BAT.SequentialExec())
        purpose_width = typemax(Int16) - 2
        step_rngpart = BAT.RNGPartition(
            Philox4x((572, 21)), Base.OneTo(6 * purpose_width),
        )
        walkerid = 17
        complement = ([2, 3, 4],)
        oracle_rng = Philox4x((572, 211))
        BAT.set_rng!(
            oracle_rng,
            BAT._mcmc_walker_rngpart(
                step_rngpart, BAT._MCMC_COMPANION_SELECTION_PURPOSE, 1,
            ),
            walkerid,
        )
        companion_idxs = only(complement)
        first_pos = rand(oracle_rng, eachindex(companion_idxs))
        second_pos = rand(oracle_rng, 1:(length(companion_idxs) - 1))
        second_pos >= first_pos && (second_pos += 1)
        companion_a, companion_b = companion_idxs[first_pos], companion_idxs[second_pos]
        BAT.set_rng!(
            oracle_rng,
            BAT._mcmc_walker_rngpart(step_rngpart, BAT._MCMC_DE_SCALE_PURPOSE, 1),
            walkerid,
        )
        gamma = proposal.gamma0 * (1 + proposal.sigma * randn(oracle_rng))
        expected = current[1] + gamma * (current[companion_a] - current[companion_b])
        candidate = fill(NaN, 2)

        BAT._propose_ensemble_candidate!!(
            candidate, proposal, current, 1, complement,
            Philox4x((572, 211)), step_rngpart, 1, walkerid,
        )
        @test candidate == expected
        @test BAT._ensemble_log_hastings(proposal, current[1], candidate, nothing) == 0
    end

    @testset "validates initialized ensemble" begin
        too_few_1d = [[-1.0], [0.0], [1.0]]
        err = _capture_ensemble_error(() -> _ensemble_move_state(DEMove(), too_few_1d))
        @test err isa ArgumentError
    end

    @testset "target failure is atomic" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        calls = Ref(0)
        target = _FailingEnsembleTarget(
            batmeasure(MvNormal(zeros(1), ones(1, 1))), x -> -sum(abs2, x) / 2,
            calls, Ref(typemax(Int)),
        )
        state = _ensemble_move_state(DEMove(sigma = 0), initial; target)
        current_before = deepcopy(state.chain_state.current)
        output_before = deepcopy(state.chain_state.output)
        calls[] = 0
        target.fail_at[] = 2

        err = _capture_ensemble_error(() -> BAT.mcmc_step!!(state))

        @test err isa BAT.EvalException
        @test state.chain_state.current == current_before
        @test state.chain_state.output == output_before
        @test state.chain_state.nattempts == [0]
        @test state.chain_state.nsamples == [0]
    end

    @testset "correlated affine Gaussian moments" begin
        # Limits are rounded 1.5 times 64-seed calibration maxima; 128
        # independent validation seeds had no joint failures per target
        # (95% one-sided upper bound 2.31%).
        _check_ensemble_gaussian_moments(
            DEMove();
            seed = 26201,
            mean_tolerance = 0.09,
            covariance_tolerance = 0.18,
        )
    end
end
