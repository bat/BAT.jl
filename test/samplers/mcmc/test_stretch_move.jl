# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using BAT: NoAdaptiveTransform, NoMCMCTempering
using DensityInterface, Distributions, LinearAlgebra, Random, Random123, ValueShapes


struct _CountingStretchTarget{M,F} <: BAT.BATMeasure
    base::M
    logdensity::F
    calls::Base.RefValue{Int}
end

DensityInterface.logdensityof(target::_CountingStretchTarget, x) =
    (target.calls[] += 1; target.logdensity(x))
ValueShapes.varshape(target::_CountingStretchTarget) = ValueShapes.varshape(target.base)


function _stretch_move_state(
    v_init;
    nwalkers = length(v_init),
    scale = 2,
    target = let
        T = float(eltype(first(v_init)))
        d = length(first(v_init))
        batmeasure(MvNormal(zeros(T, d), Diagonal(ones(T, d))))
    end,
    adaptive_transform = NoAdaptiveTransform(),
    sample_weighting = RepetitionWeighting(),
    proposal_tuning = NoMCMCProposalTuning(),
    transform_tuning = NoMCMCTransformTuning(),
)
    algorithm = TransformedMCMC(
        proposal = StretchMove(scale = scale),
        pretransform = DoNotTransform(),
        adaptive_transform = adaptive_transform,
        convergence = AssumeConvergence(),
        nwalkers = nwalkers,
        sample_weighting = sample_weighting,
        proposal_tuning = proposal_tuning,
        transform_tuning = transform_tuning,
    )
    return BAT.MCMCState(algorithm, target, 1, v_init, BATContext(rng = Philox4x((564, 80))))
end

function _capture_error(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end


function _stretch_move_oracle(state)
    oracle = deepcopy(state)
    chain_state = oracle.chain_state
    BAT.reset_rng_counters!(chain_state)

    rng = BAT.get_rng(chain_state.context)
    purpose_width = typemax(Int16) - 2
    step_rngpart = BAT.RNGPartition(rng, Base.OneTo(6 * purpose_width))
    proposal_idx = 1
    stream_idx(purpose) = (purpose - 1) * purpose_width + proposal_idx

    split_rng = Random.AbstractRNG(step_rngpart, stream_idx(4))
    logical_order = chain_state.walker_order
    permutation = randperm(split_rng, length(logical_order))
    split_at = fld(length(logical_order), 2)
    left = logical_order[permutation[begin:split_at]]
    right = logical_order[permutation[(split_at + 1):end]]
    first_group, second_group = rand(split_rng, Bool) ? (left, right) : (right, left)

    companion_stream = Random.AbstractRNG(step_rngpart, stream_idx(5))
    companion_rngpart = BAT.RNGPartition(
        companion_stream, Base.OneTo(typemax(Int32) - 2),
    )
    stretch_stream = Random.AbstractRNG(step_rngpart, stream_idx(6))
    stretch_rngpart = BAT.RNGPartition(
        stretch_stream, Base.OneTo(typemax(Int32) - 2),
    )

    initial = deepcopy(chain_state.current.z.v)
    expected = deepcopy(initial)
    stale_second = deepcopy(initial)
    scale = chain_state.proposal.scale
    for (group, complement) in ((first_group, second_group), (second_group, first_group))
        for i in group
            walkerid = chain_state.current.x.info[i].walkerid
            companion_rng = Random.AbstractRNG(companion_rngpart, walkerid)
            companion = rand(companion_rng, complement)
            stretch_rng = Random.AbstractRNG(stretch_rngpart, walkerid)
            u = rand(stretch_rng, eltype(expected[i]))
            z = ((scale - one(scale)) * u + one(scale))^2 / scale
            expected[i] = expected[companion] + z * (expected[i] - expected[companion])
            if group === second_group
                stale_second[i] = initial[companion] + z * (initial[i] - initial[companion])
            end
        end
    end

    return (; expected, stale_second, first_group, second_group)
end


function _set_walker_ids!(state, walkerids)
    chain_state = state.chain_state
    for samples in (
        chain_state.current.x, chain_state.current.z,
        chain_state.proposed.x, chain_state.proposed.z, chain_state.output,
    )
        for i in eachindex(samples.info)
            info = samples.info[i]
            samples.info[i] = BAT.MCMCSampleID(
                info.chainid, walkerids[i], info.chaincycle,
                info.stepno, info.proposalid, info.sampletype,
            )
        end
    end
    chain_state.walker_order = sortperm(walkerids)
    return state
end


@testset "StretchMove" begin
    @testset "constructor and defaults" begin
        proposal = StretchMove()
        @test proposal.scale == 2
        @test StretchMove(scale = 3f0).scale === 3f0

        for scale in (1, 0, -1, Inf, NaN)
            err = _capture_error(() -> StretchMove(scale = scale))
            @test err isa ArgumentError
            @test occursin("scale", sprint(showerror, err))
        end

        algorithm = TransformedMCMC(proposal = proposal, nwalkers = 4)
        @test algorithm.proposal_tuning isa NoMCMCProposalTuning
        @test algorithm.adaptive_transform isa NoAdaptiveTransform
        @test algorithm.transform_tuning isa NoMCMCTransformTuning
        @test algorithm.tempering isa NoMCMCTempering
        @test algorithm.init isa MCMCRetryInit
    end

    @testset "requires no adaptive transform" begin
        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        err = _capture_error(() -> _stretch_move_state(
            full_rank; adaptive_transform = BAT.TriangularAffineTransform(),
        ))
        @test err isa ArgumentError
        @test occursin("NoAdaptiveTransform", sprint(showerror, err))
    end

    @testset "requires explicit walker count" begin
        err = _capture_error(() -> TransformedMCMC(proposal = StretchMove()))
        @test err isa ArgumentError
        @test occursin("nwalkers", sprint(showerror, err))
    end

    @testset "validates initialized ensemble" begin
        too_few = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        err = _capture_error(() -> _stretch_move_state(too_few))
        @test err isa ArgumentError
        @test occursin("at least 2 * d", sprint(showerror, err))

        nearly_collinear = [[0.0, 0.0], [1.0, 1e-18], [2.0, 2e-18], [3.0, 4e-18]]
        err = _capture_error(() -> _stretch_move_state(nearly_collinear))
        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin("4 walkers", message)
        @test occursin("dimension 2", message)
        @test occursin("rank 1", message)

        nonfinite = [[0.0, 0.0], [1.0, 0.0], [0.0, NaN], [1.0, 1.0]]
        err = _capture_error(() -> _stretch_move_state(nonfinite))
        @test err isa ArgumentError
        @test occursin("finite transformed coordinates", sprint(showerror, err))

        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        @test _stretch_move_state(full_rank) isa BAT.MCMCState
        state32 = _stretch_move_state(map(v -> Float32.(v), full_rank); scale = 2.0)
        @test state32 isa BAT.MCMCState
        @test state32.chain_state.proposal.scale === 2f0

        err = _capture_error(() -> _stretch_move_state(
            map(v -> Float32.(v), full_rank); scale = nextfloat(1.0),
        ))
        @test err isa ArgumentError
        @test occursin("after conversion", sprint(showerror, err))
    end

    @testset "rejects unsupported configurations" begin
        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        err = _capture_error(() -> _stretch_move_state(full_rank; sample_weighting = ARPWeighting()))
        @test err isa ArgumentError
        @test occursin("RepetitionWeighting", sprint(showerror, err))

        err = _capture_error(() -> _stretch_move_state(
            full_rank; proposal_tuning = BAT.StepSizeAdaptor(),
        ))
        @test err isa ArgumentError
        @test occursin("NoMCMCProposalTuning", sprint(showerror, err))

        err = _capture_error(() -> _stretch_move_state(
            full_rank; transform_tuning = BAT.StanLikeTuning(),
        ))
        @test err isa ArgumentError
        @test occursin("NoMCMCTransformTuning", sprint(showerror, err))

        multi = MCMCMultiProposal(proposals = BAT.MCMCProposal[StretchMove(), RandomWalk()])
        algorithm = TransformedMCMC(
            proposal = multi,
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            convergence = AssumeConvergence(),
            nwalkers = 4,
        )
        target = batmeasure(MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)))
        err = _capture_error(() -> BAT.MCMCState(
            algorithm, target, 1, full_rank, BATContext(rng = Philox4x((564, 81))),
        ))
        @test err isa ArgumentError
        @test occursin("StretchMove", sprint(showerror, err))
    end


    @testset "proposal equation and acceptance ratio" begin
        scale = BAT._stretch_scale(2.0, 0.5)
        @test scale == 1.125
        @test BAT._stretch_candidate([2.0, 4.0], [-2.0, 0.0], 1.25) == [3.0, 5.0]
        @test BAT._stretch_log_acceptance(3, 2.0, -5.0, -7.0) ==
            3.386294361119891
    end


    @testset "full red-blue sweep" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        base = batmeasure(MvNormal(zeros(1), ones(1, 1)))
        calls = Ref(0)
        target = _CountingStretchTarget(base, _ -> 0.0, calls)
        state = _stretch_move_state(initial; target)
        oracle = _stretch_move_oracle(state)
        @test any(i -> oracle.expected[i] != oracle.stale_second[i], oracle.second_group)

        calls[] = 0
        state = BAT.mcmc_step!!(state)
        chain_state = state.chain_state

        @test chain_state.current.z.v == oracle.expected
        @test chain_state.current.x.v == oracle.expected
        @test calls[] == length(initial)
        @test chain_state.nattempts == [length(initial)]
        @test chain_state.nsamples == [length(initial)]
        @test chain_state.stepno == 1
        @test all(chain_state.accepted)
        expected_info = [
            BAT.MCMCSampleID(1, i, 1, 1, 1, true) for i in Int32(1):Int32(4)
        ]
        @test chain_state.proposed.x.info == expected_info
        @test chain_state.proposed.x.info == chain_state.proposed.z.info
    end


    @testset "subset bookkeeping synchronizes x and z" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        state = _stretch_move_state(initial)
        chain_state = state.chain_state
        chain_state.accepted .= [true, false, true, false]
        for i in eachindex(initial)
            chain_state.proposed.x.v[i] .= 10i
            chain_state.proposed.z.v[i] .= 10i
            chain_state.proposed.x.logd[i] = -i
            chain_state.proposed.z.logd[i] = -i
        end
        step_info = BAT.MCMCStepInfo([0.75, 0.0, 1.0, 0.25])

        BAT._apply_mcmc_subset!!(chain_state, step_info, 1:2)
        @test chain_state.current.x.v[1] == [10.0]
        @test chain_state.current.z.v[1] == [10.0]
        @test chain_state.current.x.v[2] == initial[2]
        @test chain_state.current.z.v[2] == initial[2]
        @test chain_state.current.x.weight == chain_state.current.z.weight
        @test chain_state.proposed.x.weight == chain_state.proposed.z.weight
        @test chain_state.current.x.weight[1:2] == [1, 1]
        @test chain_state.proposed.x.weight[1:2] == [1, 0]
        @test chain_state.output.v[1] == initial[1]
        @test chain_state.output.v[2] == [20.0]
        @test chain_state.proposed.x.info[1].sampletype
        @test !chain_state.proposed.x.info[2].sampletype
        @test all(==(0), getproperty.(chain_state.proposed.x.info[3:4], :stepno))
    end


    @testset "seed and storage-order determinism" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        repeated_a = _stretch_move_state(initial)
        repeated_b = _stretch_move_state(initial)
        repeated_a = BAT.mcmc_step!!(repeated_a)
        repeated_b = BAT.mcmc_step!!(repeated_b)
        @test repeated_a.chain_state.current.z == repeated_b.chain_state.current.z
        @test repeated_a.chain_state.proposed.z == repeated_b.chain_state.proposed.z
        @test repeated_a.chain_state.accepted == repeated_b.chain_state.accepted

        permutation = [3, 1, 4, 2]
        reordered = _stretch_move_state(initial[permutation])
        _set_walker_ids!(reordered, Int32.(permutation))
        reordered = BAT.mcmc_step!!(reordered)

        lhs_order = sortperm(getproperty.(repeated_a.chain_state.current.x.info, :walkerid))
        rhs_order = sortperm(getproperty.(reordered.chain_state.current.x.info, :walkerid))
        @test repeated_a.chain_state.current.z.v[lhs_order] ==
            reordered.chain_state.current.z.v[rhs_order]
        @test repeated_a.chain_state.proposed.z.v[lhs_order] ==
            reordered.chain_state.proposed.z.v[rhs_order]
        @test repeated_a.chain_state.accepted[lhs_order] ==
            reordered.chain_state.accepted[rhs_order]
        @test repeated_a.chain_state.walker_order == [1, 2, 3, 4]
        @test reordered.chain_state.walker_order == [2, 4, 1, 3]
    end


    @testset "generic steps preserve the post-step context RNG" begin
        algorithm = TransformedMCMC(
            proposal = RandomWalk(),
            proposal_tuning = NoMCMCProposalTuning(),
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            transform_tuning = NoMCMCTransformTuning(),
            convergence = AssumeConvergence(),
            nwalkers = 1,
        )
        target = batmeasure(MvNormal(zeros(1), ones(1, 1)))
        state = BAT.MCMCState(
            algorithm, target, 1, [[0.25]],
            BATContext(rng = Philox4x((564, 82))),
        )

        reference = deepcopy(state)
        BAT.reset_rng_counters!(reference)
        reference_rng = BAT.get_rng(reference.chain_state.context)
        purpose_width = typemax(Int16) - 2
        BAT.RNGPartition(reference_rng, Base.OneTo(3 * purpose_width))
        expected_next = rand(reference_rng)

        state = BAT.mcmc_step!!(state)
        actual_next = rand(BAT.get_rng(state.chain_state.context))
        @test actual_next == expected_next
    end


    @testset "Float32 transition values" begin
        initial = [[-3f0], [-1f0], [1f0], [4f0]]
        state = _stretch_move_state(initial; scale = 2.0)
        state = BAT.mcmc_step!!(state)
        chain_state = state.chain_state

        @test chain_state.proposal.scale isa Float32
        @test all(v -> eltype(v) === Float32, chain_state.current.z.v)
        @test all(v -> eltype(v) === Float32, chain_state.proposed.z.v)
        log_acceptance = BAT._stretch_log_acceptance(1, 1.25f0, -2f0, -3f0)
        @test log_acceptance isa Float32
        @test BAT._mcmc_acceptance_probability(log_acceptance) isa Float32
        @test BAT._mcmc_acceptance_probability(Float32(NaN)) === 0f0
    end


    @testset "non-finite candidate densities reject without state corruption" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        base = batmeasure(MvNormal(zeros(1), ones(1, 1)))
        calls = Ref(0)
        initial_values = Set(only.(initial))
        target = _CountingStretchTarget(
            base, x -> only(x) in initial_values ? 0.0 : -Inf, calls,
        )
        state = _stretch_move_state(initial; target)
        state.chain_state.current.x.logd[1] = -Inf
        state.chain_state.current.z.logd[1] = -Inf
        current_before = deepcopy(state.chain_state.current)

        calls[] = 0
        state = BAT.mcmc_step!!(state)
        chain_state = state.chain_state

        @test calls[] == length(initial)
        @test !any(chain_state.accepted)
        @test chain_state.current.x.v == current_before.x.v
        @test chain_state.current.z.v == current_before.z.v
        @test chain_state.current.x.logd == current_before.x.logd
        @test chain_state.current.z.logd == current_before.z.logd
        @test chain_state.current.x.weight == fill(1, length(initial))
        @test chain_state.current.x.weight == chain_state.current.z.weight
        @test chain_state.proposed.x.weight == chain_state.proposed.z.weight ==
            fill(0, length(initial))
    end
end
