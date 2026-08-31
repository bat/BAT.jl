# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using BAT: NoAdaptiveTransform, NoMCMCTempering
using DensityInterface, Distributions, LinearAlgebra, Random, Random123, ValueShapes


struct _CountingDESnookerTarget{M,F,C} <: BAT.BATMeasure
    base::M
    logdensity::F
    calls::C
end

_increment_de_snooker_calls!(calls::Base.RefValue{Int}) = (calls[] += 1)
_increment_de_snooker_calls!(calls::Threads.Atomic{Int}) = Threads.atomic_add!(calls, 1)

DensityInterface.logdensityof(target::_CountingDESnookerTarget, x) =
    (_increment_de_snooker_calls!(target.calls); target.logdensity(x))
ValueShapes.varshape(target::_CountingDESnookerTarget) = ValueShapes.varshape(target.base)


function _de_snooker_move_state(
    v_init;
    nwalkers = length(v_init),
    scale = 1.7,
    executor = BAT.SequentialExec(),
    target = let
        T = float(eltype(first(v_init)))
        d = length(first(v_init))
        batmeasure(MvNormal(zeros(T, d), Diagonal(ones(T, d))))
    end,
    rng_seed = (572, 30),
)
    algorithm = TransformedMCMC(
        proposal = DESnookerMove(scale = scale, executor = executor),
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


function _capture_de_snooker_error(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end


function _de_snooker_step_rngpart(state)
    BAT.reset_rng_counters!(state)
    purpose_width = typemax(Int16) - 2
    rng = BAT.get_rng(state.chain_state.context)
    return BAT.RNGPartition(rng, Base.OneTo(6 * purpose_width))
end


@testset "DESnookerMove" begin
    @testset "constructor and defaults" begin
        proposal = DESnookerMove()
        @test proposal.scale == 1.7
        @test proposal.executor isa BAT.SequentialExec
        @test DESnookerMove(scale = 1.25f0).scale === 1.25f0
        @test DESnookerMove(executor = BAT.MultiThreadedExec()).executor isa BAT.MultiThreadedExec

        for scale in (0, -1, Inf, NaN)
            err = _capture_de_snooker_error(() -> DESnookerMove(scale = scale))
            @test err isa ArgumentError
            @test occursin("scale", sprint(showerror, err))
        end
        err = _capture_de_snooker_error(() -> DESnookerMove(executor = BAT.DistributedExec()))
        @test err isa ArgumentError
        @test occursin("executor", lowercase(sprint(showerror, err)))

        algorithm = TransformedMCMC(proposal = proposal, nwalkers = 4)
        @test algorithm.proposal_tuning isa NoMCMCProposalTuning
        @test algorithm.adaptive_transform isa NoAdaptiveTransform
        @test algorithm.transform_tuning isa NoMCMCTransformTuning
        @test algorithm.tempering isa NoMCMCTempering
        @test algorithm.init isa MCMCRetryInit
    end

    @testset "proposal equation and Hastings correction" begin
        current = [3.0, 4.0]
        reference = [0.0, 0.0]
        candidate = fill(NaN, 2)
        proposal_aux = BAT._de_snooker_candidate!!(
            candidate, current, reference, [5.0, 0.0], [0.0, 5.0], 1.7,
        )
        @test candidate ≈ [1.98, 2.64]
        @test proposal_aux.direction_norm == 5.0
        @test proposal_aux.reference === reference
        proposal = BAT.DESnookerMoveProposalState(1.7, BAT.SequentialExec())
        @test BAT._ensemble_log_hastings(
            proposal, current, candidate, proposal_aux,
        ) ≈ log(0.66)
    end

    @testset "four nonempty random groups" begin
        initial = [[Float64(i), Float64(i == 1)] for i in 1:10]
        state = _de_snooker_move_state(initial)
        groups = BAT._ensemble_move_groups(
            state.chain_state.proposal, _de_snooker_step_rngpart(state),
            1, collect(eachindex(initial)),
        )
        @test length(groups) == 4
        @test all(!isempty, groups)
        @test sort(length.(groups)) == [2, 2, 3, 3]
        @test sort(reduce(vcat, groups)) == collect(eachindex(initial))
    end

    @testset "three distinct complement groups" begin
        rng = Xoshiro(0x736e6f6f6b6572)
        complement_groups = [[11, 12], [21, 22], [31, 32]]
        assignments = [
            BAT._de_snooker_companion_indices(rng, complement_groups) for _ in 1:24
        ]
        @test all(assignments) do assignment
            sort(fld.(collect(assignment), 10)) == [1, 2, 3]
        end
        @test length(unique(first.(assignments))) > 2
    end

    @testset "degenerate directions preserve the current point" begin
        proposal = BAT.DESnookerMoveProposalState(1.7, BAT.SequentialExec())
        for (current, reference) in (
            ([2.0, 3.0], [2.0, 3.0]),
            ([floatmax(Float64)], [-floatmax(Float64)]),
        )
            candidate = fill(42.0, length(current))
            proposal_aux = BAT._de_snooker_candidate!!(
                candidate, current, reference,
                zeros(length(current)), ones(length(current)), proposal.scale,
            )
            @test isnothing(proposal_aux)
            @test candidate == current
        end
    end

    @testset "degenerate directions skip target evaluation" begin
        base = batmeasure(MvNormal(zeros(1), ones(1, 1)))
        for value in (0.0, floatmax(Float64))
            calls = Ref(0)
            target = _CountingDESnookerTarget(base, _ -> 0.0, calls)
            state = _de_snooker_move_state(
                [[0.0], [1.0], [2.0], [3.0]]; target,
                rng_seed = (572, value == 0.0 ? 31 : 32),
            )
            chain_state = state.chain_state
            chain_state.current.z.v[1] .= value
            chain_state.current.x.v[1] .= value
            reference_value = iszero(value) ? value : -value
            for i in 2:4
                chain_state.current.z.v[i] .= reference_value
                chain_state.current.x.v[i] .= reference_value
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
        end
    end

    @testset "validates initialized ensemble" begin
        too_few_1d = [[-1.0], [0.0], [1.0]]
        err = _capture_de_snooker_error(() -> _de_snooker_move_state(too_few_1d))
        @test err isa ArgumentError
        @test occursin("max(2 * d, 4)", sprint(showerror, err))

        too_few_3d = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        err = _capture_de_snooker_error(() -> _de_snooker_move_state(too_few_3d))
        @test err isa ArgumentError
        @test occursin("6", sprint(showerror, err))

        rank_deficient = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        err = _capture_de_snooker_error(() -> _de_snooker_move_state(rank_deficient))
        @test err isa ArgumentError
        @test occursin("affine rank 1", sprint(showerror, err))

        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        @test _de_snooker_move_state(full_rank) isa BAT.MCMCState
        state32 = _de_snooker_move_state(map(v -> Float32.(v), full_rank))
        @test state32.chain_state.proposal.scale === 1.7f0

        tiny_scale = Float64(nextfloat(0.0f0)) / 2
        err = _capture_de_snooker_error(() -> _de_snooker_move_state(
            map(v -> Float32.(v), full_rank); scale = tiny_scale,
        ))
        @test err isa ArgumentError
        @test occursin("after conversion", sprint(showerror, err))
    end

    @testset "executor equality and target call count" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        states = map((BAT.SequentialExec(), BAT.MultiThreadedExec())) do executor
            base = batmeasure(MvNormal(zeros(1), ones(1, 1)))
            calls = Threads.Atomic{Int}(0)
            target = _CountingDESnookerTarget(base, x -> -only(x)^2 / 2, calls)
            state = _de_snooker_move_state(
                initial; executor, target, rng_seed = (572, 33),
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
end
