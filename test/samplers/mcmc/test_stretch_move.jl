# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using BAT: NoAdaptiveTransform, NoMCMCTempering
using DensityInterface, Distributions, LinearAlgebra, Random, Random123, Statistics, ValueShapes


struct _CountingStretchTarget{M,F,C} <: BAT.BATMeasure
    base::M
    logdensity::F
    calls::C
end

struct _FailingStretchTarget{M,F} <: BAT.BATMeasure
    base::M
    logdensity::F
    calls::Base.RefValue{Int}
    fail_at::Base.RefValue{Int}
end

struct _BananaStretchTarget{M,T} <: BAT.BATMeasure
    base::M
    bend::T
end

struct _StretchConvergenceRecorder <: BAT.ConvergenceTest
    calls::Vector{Any}
end

mutable struct _StretchRetryInit <: InitvalAlgorithm
    values::Vector{Vector{Float64}}
end

struct _FinalizationOldProposalState <: BAT.MCMCProposalState end
struct _FinalizationNewProposalState <: BAT.MCMCProposalState end

const _finalization_tuning_input = Ref{Any}(nothing)

function BAT.mcmc_tune_post_step!!(
    state::BAT.MCMCState,
    proposal::_FinalizationOldProposalState,
    ::BAT.MCMCStepInfo,
)
    _finalization_tuning_input[] = proposal
    return state
end

_increment_stretch_calls!(calls::Base.RefValue{Int}) = (calls[] += 1)
_increment_stretch_calls!(calls::Threads.Atomic{Int}) = Threads.atomic_add!(calls, 1)

DensityInterface.logdensityof(target::_CountingStretchTarget, x) =
    (_increment_stretch_calls!(target.calls); target.logdensity(x))
ValueShapes.varshape(target::_CountingStretchTarget) = ValueShapes.varshape(target.base)

function DensityInterface.logdensityof(target::_FailingStretchTarget, x)
    target.calls[] += 1
    target.calls[] == target.fail_at[] && error("controlled target failure")
    return target.logdensity(x)
end
ValueShapes.varshape(target::_FailingStretchTarget) = ValueShapes.varshape(target.base)

function DensityInterface.logdensityof(target::_BananaStretchTarget, x)
    latent_y = x[2] - target.bend * (x[1]^2 - one(x[1]))
    return logpdf(Normal(), x[1]) + logpdf(Normal(), latent_y)
end
ValueShapes.varshape(target::_BananaStretchTarget) = ValueShapes.varshape(target.base)

function BAT.bat_convergence_impl(
    samples::AbstractVector{<:DensitySampleVector},
    algorithm::_StretchConvergenceRecorder,
    ::BATContext,
)
    groups = map(samples) do group
        info = group.info
        return (
            chainids = sort(unique(getproperty.(info, :chainid))),
            walkerids = sort(unique(getproperty.(info, :walkerid))),
            mass = sum(group.weight),
        )
    end
    push!(algorithm.calls, groups)
    return (result = true,)
end

function BAT.bat_initval_impl(::BAT.MeasureLike, algorithm::_StretchRetryInit, ::BATContext)
    return (result = popfirst!(algorithm.values),)
end


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
    executor = BAT.SequentialExec(),
    rng_seed = (564, 80),
)
    algorithm = TransformedMCMC(
        proposal = StretchMove(scale = scale, executor = executor),
        pretransform = DoNotTransform(),
        adaptive_transform = adaptive_transform,
        convergence = AssumeConvergence(),
        nwalkers = nwalkers,
        sample_weighting = sample_weighting,
        proposal_tuning = proposal_tuning,
        transform_tuning = transform_tuning,
    )
    return BAT.MCMCState(
        algorithm, target, 1, v_init, BATContext(rng = Philox4x(rng_seed)),
    )
end

function _run_ensemble_move(
    proposal,
    target,
    v_init;
    seed::Integer,
    nwarmup::Integer,
    nsweeps::Integer,
)
    algorithm = TransformedMCMC(
        proposal = proposal,
        pretransform = DoNotTransform(),
        adaptive_transform = NoAdaptiveTransform(),
        convergence = AssumeConvergence(),
        nwalkers = length(v_init),
        sample_weighting = RepetitionWeighting(),
        proposal_tuning = NoMCMCProposalTuning(),
        transform_tuning = NoMCMCTransformTuning(),
    )
    state = BAT.MCMCState(
        algorithm, target, 1, v_init,
        BATContext(rng = Philox4x((564, seed))),
    )
    state = BAT.mcmc_iterate!!(nothing, state; max_nsteps = nwarmup)
    outputs = BAT._empty_chain_outputs(state)
    state = BAT.mcmc_iterate!!(outputs, state; max_nsteps = nsweeps)
    samples = BAT._merge_chain_outputs(state, [outputs])
    return (; state, outputs, samples)
end

_run_stretch_move(target, v_init; kwargs...) =
    _run_ensemble_move(StretchMove(), target, v_init; kwargs...)

function _trace_stretch_move(target, v_init; seed::Integer, nsweeps::Integer)
    state = _stretch_move_state(v_init; target, rng_seed = (564, seed))
    outputs = BAT._empty_chain_outputs(state)
    accepted = BitMatrix(undef, length(v_init), nsweeps)
    for step in 1:nsweeps
        state = BAT.mcmc_iterate!!(outputs, state; max_nsteps = 1)
        accepted[:, step] = state.chain_state.accepted
    end
    samples = BAT._merge_chain_outputs(state, [outputs])
    return (; state, outputs, samples, accepted)
end

function _two_dimensional_elliptic_initial_ensemble(
    mean,
    covariance,
    nwalkers::Integer;
    radius = 1.5,
)
    scale = cholesky(Symmetric(covariance)).L
    return [
        mean + scale * [
            radius * cospi(2 * k / nwalkers),
            radius * sinpi(2 * k / nwalkers),
        ]
        for k in 0:(nwalkers - 1)
    ]
end

_banana_point(z, bend) = [z[1], z[2] + bend * (z[1]^2 - one(z[1]))]

function _check_ensemble_gaussian_moments(
    proposal;
    seeds,
    tolerances,
)
    cases = (
        (
            name = "standard Gaussian",
            mean = [0.0, 0.0],
            covariance = [1.0 0.0; 0.0 1.0],
            seed = seeds.standard,
            mean_tolerance = tolerances.standard_mean,
            covariance_tolerance = tolerances.standard_covariance,
        ),
        (
            name = "affine Gaussian",
            mean = [0.7, -1.2],
            covariance = [1.4 0.8; 0.8 2.0],
            seed = seeds.affine,
            mean_tolerance = tolerances.affine_mean,
            covariance_tolerance = tolerances.affine_covariance,
        ),
    )

    for case in cases
        @testset "$(nameof(typeof(proposal))) $(case.name)" begin
            nwalkers = 16
            initial = _two_dimensional_elliptic_initial_ensemble(
                case.mean, case.covariance, nwalkers; radius = 0.5,
            )
            target = batmeasure(MvNormal(case.mean, case.covariance))
            result = _run_ensemble_move(
                proposal, target, initial;
                seed = case.seed,
                nwarmup = 500,
                nsweeps = 1500,
            )

            @test sum(result.samples.weight) == nwalkers * 1500
            @test maximum(abs, mean(result.samples) - case.mean) <
                case.mean_tolerance
            @test maximum(abs, cov(result.samples) - case.covariance) <
                case.covariance_tolerance
        end
    end
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
    # Keep the stream numbering independent to catch production-layout changes.
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
        @test proposal.executor isa BAT.SequentialExec
        @test StretchMove(scale = 3f0).scale === 3f0
        @test StretchMove(executor = BAT.MultiThreadedExec()).executor isa BAT.MultiThreadedExec

        for scale in (1, 0, -1, Inf, NaN)
            err = _capture_error(() -> StretchMove(scale = scale))
            @test err isa ArgumentError
            @test occursin("scale", sprint(showerror, err))
        end
        err = _capture_error(() -> StretchMove(executor = BAT.DistributedExec()))
        @test err isa ArgumentError
        @test occursin("executor", lowercase(sprint(showerror, err)))

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

        target = batmeasure(MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)))
        context = BATContext(rng = Philox4x((564, 86)))
        converted_scale = try
            BAT._create_proposal_state(
                StretchMove(scale = 2.0),
                target,
                context,
                full_rank,
                map(v -> Float32.(v), full_rank),
                identity,
                BAT.get_rng(context),
            ).scale
        catch
            nothing
        end
        @test converted_scale === 2f0

        err = _capture_error(() -> _stretch_move_state(
            map(v -> Float32.(v), full_rank); scale = nextfloat(1.0),
        ))
        @test err isa ArgumentError
        @test occursin("after conversion", sprint(showerror, err))
    end

    @testset "retry initialization preserves affine rank" begin
        initial = [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 1.0]]
        initval_alg = _StretchRetryInit([initial; [[0.4, 0.0]]])
        init_alg = MCMCRetryInit(
            max_init_tries = 2,
            nsteps_init = 1,
            initval_alg = initval_alg,
        )
        algorithm = TransformedMCMC(
            proposal = StretchMove(),
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            transform_tuning = NoMCMCTransformTuning(),
            init = init_alg,
            convergence = AssumeConvergence(),
            nchains = 1,
            nwalkers = length(initial),
        )
        target = batmeasure(product_distribution([
            Uniform(0, 1), Uniform(-1e-12, 1e-12),
        ]))

        err = _capture_error(() -> BAT.mcmc_init!(
            algorithm,
            target,
            init_alg,
            (args...) -> nothing,
            BATContext(rng = Philox4x((564, 85))),
        ))

        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin("4 walkers", message)
        @test occursin("dimension 2", message)
        @test occursin("rank 1", message)
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

    end

    @testset "weighted ensemble mixtures" begin
        full_rank = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        target = batmeasure(MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)))
        proposal = MCMCMultiProposal(
            proposals = BAT.MCMCProposal[StretchMove(), DEMove(gamma0 = 0.5, sigma = 0)],
            picking_rule = [2, 1],
        )
        algorithm = TransformedMCMC(
            proposal = proposal,
            pretransform = DoNotTransform(),
            convergence = AssumeConvergence(),
            nwalkers = 4,
        )

        @test algorithm.adaptive_transform isa NoAdaptiveTransform
        @test algorithm.transform_tuning isa NoMCMCTransformTuning
        @test algorithm.sample_weighting isa RepetitionWeighting

        state = BAT.MCMCState(
            algorithm, target, 1, full_rank,
            BATContext(rng = Philox4x((564, 91))),
        )
        selected = Int[]
        proposal_ids = Vector{Vector{Int32}}()
        for _ in 1:6
            state = BAT.mcmc_step!!(state)
            push!(selected, state.chain_state.proposal.active_idx)
            push!(proposal_ids, getproperty.(state.chain_state.proposed.x.info, :proposalid))
        end
        @test selected == [1, 1, 2, 1, 1, 2]
        @test proposal_ids == [fill(Int32(i), 4) for i in [1, 1, 2, 1, 1, 2]]
        @test state.chain_state.nattempts == [16, 8]

        invalid_weighting = TransformedMCMC(
            proposal = proposal,
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            transform_tuning = NoMCMCTransformTuning(),
            convergence = AssumeConvergence(),
            nwalkers = 4,
            sample_weighting = ARPWeighting(),
        )
        err = _capture_error(() -> BAT.MCMCState(
            invalid_weighting, target, 1, full_rank,
            BATContext(rng = Philox4x((564, 92))),
        ))
        @test err isa ArgumentError
        @test occursin("RepetitionWeighting", sprint(showerror, err))

        invalid_adaptive_transform = TransformedMCMC(
            proposal = proposal,
            pretransform = DoNotTransform(),
            adaptive_transform = BAT.TriangularAffineTransform(),
            transform_tuning = NoMCMCTransformTuning(),
            convergence = AssumeConvergence(),
            nwalkers = 4,
        )
        err = _capture_error(() -> BAT.MCMCState(
            invalid_adaptive_transform, target, 1, full_rank,
            BATContext(rng = Philox4x((564, 93))),
        ))
        @test err isa ArgumentError
        @test occursin("NoAdaptiveTransform", sprint(showerror, err))

        invalid_transform_tuning = TransformedMCMC(
            proposal = proposal,
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            transform_tuning = BAT.StanLikeTuning(),
            convergence = AssumeConvergence(),
            nwalkers = 4,
        )
        err = _capture_error(() -> BAT.MCMCState(
            invalid_transform_tuning, target, 1, full_rank,
            BATContext(rng = Philox4x((564, 94))),
        ))
        @test err isa ArgumentError
        @test occursin("NoMCMCTransformTuning", sprint(showerror, err))

        rank_deficient = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        err = _capture_error(() -> BAT.MCMCState(
            algorithm, target, 1, rank_deficient,
            BATContext(rng = Philox4x((564, 95))),
        ))
        @test err isa ArgumentError
        @test occursin("affine rank 1", sprint(showerror, err))
    end

    @testset "keeps generic transform tuning compatible" begin
        algorithm = TransformedMCMC(
            proposal = RandomWalk(),
            pretransform = DoNotTransform(),
            adaptive_transform = BAT.TriangularAffineTransform(),
            transform_tuning = BAT.RAMTuning(),
            convergence = AssumeConvergence(),
            nwalkers = 1,
        )
        target = batmeasure(MvNormal(zeros(2), Matrix{Float64}(I, 2, 2)))
        @test BAT.MCMCState(
            algorithm, target, 1, [[0.0, 0.0]], BATContext(rng = Philox4x((564, 82))),
        ) isa BAT.MCMCState
    end


    @testset "proposal equation and acceptance ratio" begin
        scale = BAT._stretch_scale(2.0, 0.5)
        @test scale == 1.125
        @test BAT._stretch_scale(2.0, 0.0) == 0.5
        @test BAT._stretch_scale(2.0, 1.0) == 2.0
        @test BAT._stretch_scale(2f0, 0.5f0) === 1.125f0
        for T in (Float32, Float64)
            large_scale = floatmax(T)
            stretch_factors = BAT._stretch_scale.(large_scale, T[0, 0.5, 1])
            @test all(isfinite, stretch_factors)
            @test stretch_factors[begin] == inv(large_scale)
            @test stretch_factors[end] == large_scale
        end
        candidate = fill(NaN, 2)
        @test BAT._stretch_candidate!!(
            candidate, [2.0, 4.0], [-2.0, 0.0], 1.25,
        ) === candidate
        @test candidate == [3.0, 5.0]
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


    @testset "exact sequential sweep regression" begin
        state = _stretch_move_state(
            [[-3.0], [-1.0], [1.0], [4.0]];
            executor = BAT.SequentialExec(),
        )

        state = BAT.mcmc_step!!(state)
        chain_state = state.chain_state

        @test chain_state.current.z.v == [[-3.0], [-1.0], [1.0], [2.049883476974204]]
        @test chain_state.current.x.v == chain_state.current.z.v
        @test chain_state.current.x.logd == [
            -5.418938533204673,
            -1.4189385332046727,
            -1.4189385332046727,
            -3.019949667790599,
        ]
        @test chain_state.proposed.z.v == [
            [-5.533480715431719],
            [-2.470533404283069],
            [2.720217374950521],
            [2.049883476974204],
        ]
        @test chain_state.proposed.x.logd == [
            -16.22864294723204,
            -3.9707061840439173,
            -4.618729816696025,
            -3.019949667790599,
        ]
        @test chain_state.accepted == [false, false, false, true]
        @test chain_state.current.x.weight == [1, 1, 1, 1]
        @test chain_state.proposed.x.weight == [0, 0, 0, 1]
        @test chain_state.proposed.x.info == [
            BAT.MCMCSampleID(1, i, 1, 1, 1, i == 4) for i in Int32(1):Int32(4)
        ]
        @test chain_state.nattempts == [4]
        @test chain_state.nsamples == [1]
        @test chain_state.stepno == 1
    end


    @testset "executor equality and target call count" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        states = map((BAT.SequentialExec(), BAT.MultiThreadedExec())) do executor
            base = batmeasure(MvNormal(zeros(1), ones(1, 1)))
            calls = Threads.Atomic{Int}(0)
            target = _CountingStretchTarget(base, x -> -only(x)^2 / 2, calls)
            state = _stretch_move_state(initial; target, executor)
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


    @testset "target failure does not commit an active group" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        base = batmeasure(MvNormal(zeros(1), ones(1, 1)))
        calls = Ref(0)
        fail_at = Ref(typemax(Int))
        target = _FailingStretchTarget(base, _ -> 0.0, calls, fail_at)
        state = _stretch_move_state(initial; target, executor = BAT.SequentialExec())
        current_before = deepcopy(state.chain_state.current)
        output_before = deepcopy(state.chain_state.output)
        calls[] = 0
        fail_at[] = 2

        err = _capture_error(() -> BAT.mcmc_step!!(state))

        @test err isa BAT.EvalException
        @test occursin("controlled target failure", sprint(showerror, err))
        @test state.chain_state.current == current_before
        @test state.chain_state.output == output_before
        @test state.chain_state.nattempts == [0]
        @test state.chain_state.nsamples == [0]
    end

    @testset "large finite scale keeps sweep candidates finite" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        base = batmeasure(MvNormal(zeros(1), ones(1, 1)))
        calls = Ref(0)
        initial_values = Set(only.(initial))
        target = _CountingStretchTarget(
            base, x -> only(x) in initial_values ? 0.0 : -Inf, calls,
        )
        state = _stretch_move_state(initial; target, scale = 1e200)

        calls[] = 0
        state = BAT.mcmc_step!!(state)

        @test calls[] == length(initial)
        @test all(v -> all(isfinite, v), state.chain_state.proposed.z.v)
        @test all(v -> all(isfinite, v), state.chain_state.current.z.v)
        @test !any(state.chain_state.accepted)
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

    @testset "finalization retains the pre-update tuning state" begin
        algorithm = TransformedMCMC(
            proposal = MCMCMultiProposal(
                proposals = BAT.MCMCProposal[RandomWalk(), RandomWalk()],
                picking_rule = [1, 0],
            ),
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            transform_tuning = NoMCMCTransformTuning(),
            convergence = AssumeConvergence(),
            nwalkers = 1,
        )
        target = batmeasure(MvNormal(zeros(1), ones(1, 1)))
        state = BAT.MCMCState(
            algorithm, target, 1, [[0.25]], BATContext(rng = Philox4x((564, 83))),
        )
        active_proposal = _FinalizationOldProposalState()
        active_proposal_new = _FinalizationNewProposalState()
        _finalization_tuning_input[] = nothing

        state = BAT._finalize_mcmc_step!!(
            state, active_proposal, active_proposal_new, BAT.MCMCStepInfo([1.0]),
        )

        @test _finalization_tuning_input[] === active_proposal
        @test BAT.get_active_proposal(state.chain_state.proposal) === active_proposal_new
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


    @testset "proposal-aware ensemble ESS" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        state = _stretch_move_state(initial)
        outputs = BAT._empty_chain_outputs(state)
        state = BAT.mcmc_iterate!!(outputs, state; max_nsteps = 32)
        merged = BAT._merge_chain_outputs(state, [outputs])

        ess = BAT._mcmc_ess(
            [outputs], merged, StretchMove(),
            RepetitionWeighting(), false, BATContext(),
        )
        @test ess isa Real
        @test 0 < ess <= sum(merged.weight)
        mixture = MCMCMultiProposal(
            proposals = BAT.MCMCProposal[StretchMove(), RandomWalk()],
            picking_rule = [1, 1],
        )
        @test BAT._mcmc_ess(
            [outputs], merged, mixture,
            RepetitionWeighting(), false, BATContext(),
        ) == ess
        @test isnothing(BAT._mcmc_ess(
            [outputs], merged, StretchMove(),
            RepetitionWeighting(), true, BATContext(),
        ))
    end


    @testset "reports per-ensemble acceptance diagnostics" begin
        nwalkers = 4
        nsteps = 16
        algorithm = TransformedMCMC(
            proposal = StretchMove(),
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            transform_tuning = NoMCMCTransformTuning(),
            init = MCMCRetryInit(max_init_tries = 2, nsteps_init = 1),
            burnin = MCMCMultiCycleBurnin(
                nsteps_per_cycle = 1,
                max_ncycles = 1,
                nsteps_final = 0,
            ),
            convergence = AssumeConvergence(),
            nchains = 2,
            nwalkers = nwalkers,
            nsteps = nsteps,
        )
        evaluated = evalmeasure(
            batmeasure(Normal()),
            algorithm,
            BATContext(rng = Philox4x((564, 87))),
        )
        diagnostics = get(BAT.evalinfo(evaluated).result, :chain_diagnostics, nothing)

        @test diagnostics isa AbstractVector
        if diagnostics isa AbstractVector
            @test length(diagnostics) == algorithm.nchains
            @test all(d -> d.cycle_n_attempts == nwalkers * nsteps, diagnostics)
            @test all(d -> 0 <= d.cycle_n_accepted <= d.cycle_n_attempts, diagnostics)
            @test all(
                d -> d.cycle_acceptance_rate ==
                    d.cycle_n_accepted / d.cycle_n_attempts,
                diagnostics,
            )
        end

        zero_state = _stretch_move_state([[-3.0], [-1.0], [1.0], [4.0]])
        zero_summary = BAT._mcmc_diagnostics_summary([zero_state])
        zero_diagnostics = get(zero_summary, :chain_diagnostics, nothing)
        @test zero_diagnostics isa AbstractVector
        if zero_diagnostics isa AbstractVector
            zero_diag = only(zero_diagnostics)
            @test zero_diag.cycle_n_attempts == 0
            @test zero_diag.cycle_n_accepted == 0
            @test isnan(zero_diag.cycle_acceptance_rate)
        end
    end


    @testset "burn-in without proposal tuning" begin
        initial = [[-3.0], [-1.0], [1.0], [4.0]]
        algorithm = TransformedMCMC(
            proposal = StretchMove(),
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            convergence = AssumeConvergence(),
            nwalkers = length(initial),
            proposal_tuning = NoMCMCProposalTuning(),
            transform_tuning = NoMCMCTransformTuning(),
            burnin = MCMCMultiCycleBurnin(
                nsteps_per_cycle = 8,
                max_ncycles = 1,
                nsteps_final = 0,
            ),
        )
        target = batmeasure(MvNormal(zeros(1), ones(1, 1)))
        state = BAT.MCMCState(
            algorithm,
            target,
            1,
            initial,
            BATContext(rng = Philox4x((564, 83))),
        )

        states = BAT.mcmc_burnin!(nothing, [state], algorithm, (args...) -> nothing)

        @test only(states).chain_state.info.tuned
        @test only(states).chain_state.info.converged
    end


    @testset "Gaussian stationary moments" begin
        # Each limit is a rounded 1.5 times the largest error from 64
        # calibration seeds. None of 128 independent validation seeds failed
        # either joint mean/covariance check (95% one-sided upper bound 2.31%).
        _check_ensemble_gaussian_moments(
            StretchMove();
            seeds = (standard = 25101, affine = 25201),
            tolerances = (
                standard_mean = 0.16,
                standard_covariance = 0.15,
                affine_mean = 0.18,
                affine_covariance = 0.22,
            ),
        )
    end


    @testset "paired affine equivariance" begin
        target_mean = [0.7, -1.2]
        target_cov = [1.4 0.8; 0.8 2.0]
        affine_matrix = [1.3 0.4; -0.2 0.8]
        affine_shift = [-0.6, 1.1]
        nwalkers = 16
        initial = _two_dimensional_elliptic_initial_ensemble(target_mean, target_cov, nwalkers)
        transformed_initial = [affine_matrix * x + affine_shift for x in initial]
        target = batmeasure(MvNormal(target_mean, target_cov))
        transformed_target = batmeasure(MvNormal(
            affine_matrix * target_mean + affine_shift,
            Symmetric(affine_matrix * target_cov * affine_matrix'),
        ))

        reference = _trace_stretch_move(target, initial; seed = 4313, nsweeps = 64)
        transformed = _trace_stretch_move(
            transformed_target,
            transformed_initial;
            seed = 4313,
            nsweeps = 64,
        )
        mapped_back = [
            affine_matrix \ (y - affine_shift) for y in transformed.samples.v
        ]

        @test length(reference.samples) == length(transformed.samples)
        @test all(
            isapprox(x, y; atol = 1e-10, rtol = 0)
            for (x, y) in zip(reference.samples.v, mapped_back)
        )
        @test reference.samples.weight == transformed.samples.weight
        @test reference.accepted == transformed.accepted
        @test reference.state.chain_state.nsamples == transformed.state.chain_state.nsamples
        @test reference.state.chain_state.nattempts == transformed.state.chain_state.nattempts
    end


    @testset "analytic banana moments" begin
        bend = 0.35
        target_mean = zeros(2)
        target_cov = [1.0 0.0; 0.0 1 + 2 * bend^2]
        target_mixed_moment = 2 * bend
        nwalkers = 16
        latent_initial = _two_dimensional_elliptic_initial_ensemble(zeros(2), Matrix{Float64}(I, 2, 2), nwalkers)
        initial = [_banana_point(z, bend) for z in latent_initial]
        target = _BananaStretchTarget(
            batmeasure(MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))), bend,
        )

        result = _run_stretch_move(
            target,
            initial;
            seed = 4222,
            nwarmup = 500,
            nsweeps = 1500,
        )
        sample_mean = mean(result.samples)
        sample_cov = cov(result.samples)
        sample_mixed_moment = sum(
            result.samples.weight .* [x[1]^2 * x[2] for x in result.samples.v],
        ) / sum(result.samples.weight)

        @test maximum(abs, sample_mean - target_mean) < 0.12
        @test maximum(abs, sample_cov - target_cov) < 0.13
        @test abs(sample_mixed_moment - target_mixed_moment) < 0.27
    end


    @testset "weighted-mixture analytic banana moments" begin
        bend = 0.35
        target_mean = zeros(2)
        target_cov = [1.0 0.0; 0.0 1 + 2 * bend^2]
        target_mixed_moment = 2 * bend
        nwalkers = 16
        latent_initial = _two_dimensional_elliptic_initial_ensemble(
            zeros(2), Matrix{Float64}(I, 2, 2), nwalkers; radius = 0.5,
        )
        initial = [_banana_point(z, bend) for z in latent_initial]
        target = _BananaStretchTarget(
            batmeasure(MvNormal(zeros(2), Matrix{Float64}(I, 2, 2))), bend,
        )
        proposal = MCMCMultiProposal(
            proposals = BAT.MCMCProposal[
                StretchMove(), DEMove(), DESnookerMove(),
            ],
            picking_rule = [4, 2, 1],
        )

        result = _run_ensemble_move(
            proposal, target, initial;
            seed = 28101,
            nwarmup = 750,
            nsweeps = 2000,
        )
        sample_mean = mean(result.samples)
        sample_cov = cov(result.samples)
        sample_mixed_moment = sum(
            result.samples.weight .* [x[1]^2 * x[2] for x in result.samples.v],
        ) / sum(result.samples.weight)

        @test sum(result.samples.weight) == nwalkers * 2000
        @test all(>(0), result.state.chain_state.nattempts)
        # The three limits use the same 64-seed calibration and independent
        # 128-seed validation rule as the Gaussian checks (0 joint failures;
        # 95% one-sided upper bound 2.31%).
        @test maximum(abs, sample_mean - target_mean) < 0.11
        @test maximum(abs, sample_cov - target_cov) < 0.11
        @test abs(sample_mixed_moment - target_mixed_moment) < 0.25
    end


    @testset "convergence groups independent ensembles" begin
        target_mean = [0.7, -1.2]
        target_cov = [1.4 0.8; 0.8 2.0]
        target = batmeasure(MvNormal(target_mean, target_cov))
        nwalkers = 8
        nsteps = 8
        initial = _two_dimensional_elliptic_initial_ensemble(target_mean, target_cov, nwalkers)
        recorder = _StretchConvergenceRecorder(Any[])
        algorithm = TransformedMCMC(
            proposal = StretchMove(),
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            convergence = recorder,
            nwalkers = nwalkers,
            proposal_tuning = NoMCMCProposalTuning(),
            transform_tuning = NoMCMCTransformTuning(),
            burnin = MCMCMultiCycleBurnin(
                nsteps_per_cycle = nsteps,
                max_ncycles = 1,
                nsteps_final = 0,
            ),
        )
        states = [
            BAT.MCMCState(
                algorithm,
                target,
                chainid,
                initial,
                BATContext(rng = Philox4x((564, 4400 + chainid))),
            )
            for chainid in 1:2
        ]

        states = BAT.mcmc_burnin!(nothing, states, algorithm, (args...) -> nothing)
        groups = only(recorder.calls)

        @test length(groups) == 2
        @test only.(getproperty.(groups, :chainids)) == Int32[1, 2]
        @test all(group -> group.walkerids == Int32.(1:nwalkers), groups)
        @test all(group -> group.mass == nwalkers * nsteps, groups)
        @test all(state -> state.chain_state.info.tuned, states)
        @test all(state -> state.chain_state.info.converged, states)
    end
end
