# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using BAT: NoAdaptiveTransform, NoMCMCTempering
using DensityInterface, Distributions, LinearAlgebra, Random, Random123, ValueShapes


struct _CountingDETarget{M,F,C} <: BAT.BATMeasure
    base::M
    logdensity::F
    calls::C
end

_increment_de_calls!(calls::Base.RefValue{Int}) = (calls[] += 1)
_increment_de_calls!(calls::Threads.Atomic{Int}) = Threads.atomic_add!(calls, 1)

DensityInterface.logdensityof(target::_CountingDETarget, x) =
    (_increment_de_calls!(target.calls); target.logdensity(x))
ValueShapes.varshape(target::_CountingDETarget) = ValueShapes.varshape(target.base)


function _de_move_state(
    v_init;
    nwalkers = length(v_init),
    gamma0 = nothing,
    sigma = 1e-5,
    executor = BAT.SequentialExec(),
    target = let
        T = float(eltype(first(v_init)))
        d = length(first(v_init))
        batmeasure(MvNormal(zeros(T, d), Diagonal(ones(T, d))))
    end,
    rng_seed = (572, 20),
)
    algorithm = TransformedMCMC(
        proposal = DEMove(gamma0 = gamma0, sigma = sigma, executor = executor),
        pretransform = DoNotTransform(),
        adaptive_transform = NoAdaptiveTransform(),
        convergence = AssumeConvergence(),
        nwalkers = nwalkers,
        sample_weighting = RepetitionWeighting(),
        proposal_tuning = NoMCMCProposalTuning(),
        transform_tuning = NoMCMCTransformTuning(),
    )
    return BAT.MCMCState(
        algorithm, target, 1, v_init,
        BATContext(rng = Philox4x(rng_seed)),
    )
end


function _capture_de_error(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end


@testset "DEMove" begin
    @testset "constructor and defaults" begin
        proposal = DEMove()
        @test isnothing(proposal.gamma0)
        @test proposal.sigma == 1e-5
        @test proposal.executor isa BAT.SequentialExec
        @test DEMove(gamma0 = 0.8f0).gamma0 === 0.8f0
        @test DEMove(sigma = 0f0).sigma === 0f0
        @test DEMove(executor = BAT.MultiThreadedExec()).executor isa BAT.MultiThreadedExec

        for gamma0 in (0, -1, Inf, NaN)
            err = _capture_de_error(() -> DEMove(gamma0 = gamma0))
            @test err isa ArgumentError
            @test occursin("gamma0", sprint(showerror, err))
        end
        for sigma in (-1, Inf, NaN)
            err = _capture_de_error(() -> DEMove(sigma = sigma))
            @test err isa ArgumentError
            @test occursin("sigma", sprint(showerror, err))
        end
        err = _capture_de_error(() -> DEMove(executor = BAT.DistributedExec()))
        @test err isa ArgumentError
        @test occursin("executor", lowercase(sprint(showerror, err)))

        algorithm = TransformedMCMC(proposal = proposal, nwalkers = 4)
        @test algorithm.proposal_tuning isa NoMCMCProposalTuning
        @test algorithm.adaptive_transform isa NoAdaptiveTransform
        @test algorithm.transform_tuning isa NoMCMCTransformTuning
        @test algorithm.tempering isa NoMCMCTempering
        @test algorithm.init isa MCMCRetryInit
    end

    @testset "proposal equation" begin
        candidate = fill(NaN, 2)
        @test BAT._de_candidate!!(
            candidate, [2.0, 4.0], [6.0, -2.0], [-2.0, 2.0], 0.25,
        ) === candidate
        @test candidate == [4.0, 3.0]
        @test BAT._ensemble_log_hastings(
            BAT.DEMoveProposalState(0.75, 0.0, BAT.SequentialExec()),
            [2.0, 4.0], candidate, nothing,
        ) == 0.0
    end

    @testset "default scale and distinct ordered companions" begin
        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        state = _de_move_state(full_rank)
        @test state.chain_state.proposal.gamma0 == 2.38 / sqrt(2 * 2)

        rng = Xoshiro(0x44454d6f7665)
        pairs = [BAT._de_companion_indices(rng, [7, 11]) for _ in 1:16]
        @test all(pair -> pair[1] != pair[2], pairs)
        @test Set(pairs) == Set([(7, 11), (11, 7)])
    end

    @testset "validates initialized ensemble" begin
        too_few_1d = [[-1.0], [0.0], [1.0]]
        err = _capture_de_error(() -> _de_move_state(too_few_1d))
        @test err isa ArgumentError
        @test occursin("max(2 * d, 4)", sprint(showerror, err))

        too_few_3d = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        err = _capture_de_error(() -> _de_move_state(too_few_3d))
        @test err isa ArgumentError
        @test occursin("6", sprint(showerror, err))

        rank_deficient = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        err = _capture_de_error(() -> _de_move_state(rank_deficient))
        @test err isa ArgumentError
        @test occursin("affine rank 1", sprint(showerror, err))

        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        @test _de_move_state(full_rank) isa BAT.MCMCState
        state32 = _de_move_state(map(v -> Float32.(v), full_rank))
        @test state32.chain_state.proposal.gamma0 isa Float32
        @test state32.chain_state.proposal.sigma isa Float32
    end

    @testset "executor equality and target call count" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        states = map((BAT.SequentialExec(), BAT.MultiThreadedExec())) do executor
            base = batmeasure(MvNormal(zeros(1), ones(1, 1)))
            calls = Threads.Atomic{Int}(0)
            target = _CountingDETarget(base, x -> -only(x)^2 / 2, calls)
            state = _de_move_state(
                initial; sigma = 1e-3, executor, target, rng_seed = (572, 22),
            )
            calls[] = 0
            state = BAT.mcmc_step!!(state)
            @test calls[] == length(initial)
            return state
        end

        sequential = first(states).chain_state
        threaded = last(states).chain_state
        @test sequential.current == threaded.current
        @test sequential.proposed == threaded.proposed
        @test sequential.output == threaded.output
        @test sequential.accepted == threaded.accepted
        @test sequential.nattempts == threaded.nattempts
        @test sequential.nsamples == threaded.nsamples
    end

    @testset "categorical mixed-proposal sweeps" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        calls = Ref(0)
        base = batmeasure(MvNormal(zeros(1), ones(1, 1)))
        target = _CountingDETarget(base, x -> -only(x)^2 / 2, calls)
        proposal = MCMCMultiProposal(
            proposals = BAT.MCMCProposal[
                DEMove(gamma0 = 0.5, sigma = 0),
                RandomWalk(),
            ],
            picking_rule = Categorical([0.75, 0.25]),
        )
        algorithm = TransformedMCMC(
            proposal = proposal,
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            transform_tuning = NoMCMCTransformTuning(),
            convergence = AssumeConvergence(),
            nwalkers = length(initial),
            sample_weighting = RepetitionWeighting(),
        )
        state = BAT.MCMCState(
            algorithm, target, 1, initial,
            BATContext(rng = Philox4x((572, 30))),
        )

        calls[] = 0
        selected = zeros(Int, 2)
        for _ in 1:400
            state = BAT.mcmc_step!!(state)
            active_idx = state.chain_state.proposal.active_idx
            selected[active_idx] += 1
            @test getproperty.(state.chain_state.proposed.x.info, :proposalid) ==
                fill(Int32(active_idx), length(initial))
        end

        @test 270 <= selected[1] <= 330
        @test selected[2] == 400 - selected[1]
        @test state.chain_state.nattempts == length(initial) .* selected
        @test calls[] == 400 * length(initial)
    end
end
