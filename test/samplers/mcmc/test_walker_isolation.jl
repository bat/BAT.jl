# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module MCMCWalkerIsolationTests

using BAT
using Test

using ArraysOfArrays, DensityInterface, Distributions, LinearAlgebra, Random, Random123
import ForwardDiff

using BAT: NoAdaptiveTransform, NoMCMCProposalTuning, NoMCMCTransformTuning


struct _FailureHandshake
    sibling_started::Channel{Nothing}
    failing_task::Channel{Task}
    sibling_steps::Threads.Atomic{Int}
end

_FailureHandshake() = _FailureHandshake(
    Channel{Nothing}(0), Channel{Task}(0), Threads.Atomic{Int}(0),
)

struct _CausalTarget <: ContinuousMultivariateDistribution
    handshake::_FailureHandshake
    armed::Threads.Atomic{Bool}
end
Base.length(::_CausalTarget) = 2
Base.size(::_CausalTarget) = (2,)

function Distributions._logpdf(target::_CausalTarget, x::AbstractVector)
    if target.armed[]
        put!(target.handshake.failing_task, current_task())
        error("walker-isolation target failure")
    end
    return -sum(abs2, x) / 2
end
struct _ScriptedInit <: InitvalAlgorithm
    calls::Threads.Atomic{Int}
end

function BAT.bat_initval_impl(::BAT.MeasureLike, algorithm::_ScriptedInit, ::BATContext)
    call = Threads.atomic_add!(algorithm.calls, 1) + 1
    return (result = call == 1 ? fill(2.0, 2) : zeros(2),)
end
struct _ViabilityWeighting <: BAT.AbstractMCMCWeightingScheme{Float64} end

BAT.mcmc_weight_type(::_ViabilityWeighting) = Float64

function BAT.mcmc_weight_values(::_ViabilityWeighting, p_accept, accepted)
    viable = p_accept .> 0
    return Float64.(viable .& .!accepted), Float64.(viable .& accepted)
end
struct _FailingOutputs{T} <: AbstractVector{T}
    handshake::_FailureHandshake
end

Base.size(::_FailingOutputs) = (1,)
Base.getindex(outputs::_FailingOutputs, ::Int) = begin
    put!(outputs.handshake.failing_task, current_task())
    error("walker-isolation output failure")
end
struct _FixedTransitionProposal <: BAT.MCMCProposal
    extra_draws::Int
end
struct _FixedTransitionProposalState <: BAT.SimpleMCMCProposalState
    extra_draws::Int
end

BAT._create_proposal_state(
    proposal::_FixedTransitionProposal,
    ::BAT.BATMeasure,
    ::BATContext,
    ::AbstractVector,
    ::Function,
    ::AbstractRNG,
) = _FixedTransitionProposalState(proposal.extra_draws)

function BAT.mcmc_propose_transition(
    current_z::ArrayOfSimilarArrays,
    proposal::_FixedTransitionProposalState,
    genctxs::AbstractVector,
)
    innovation = batmeasure(Normal())
    for genctx in genctxs, _ in 1:proposal.extra_draws
        rand(genctx, innovation)
    end
    transition = [fill(0.75, length(z)) for z in current_z]
    return current_z .+ transition, zeros(length(current_z))
end
struct _CausalProposal <: BAT.MCMCProposal
    role::Symbol
    handshake::_FailureHandshake
end

struct _CausalProposalState <: BAT.SimpleMCMCProposalState
    proposal::_CausalProposal
end

BAT._create_proposal_state(
    proposal::_CausalProposal,
    ::BAT.BATMeasure,
    ::BAT.BATContext,
    ::AbstractVector,
    ::Function,
    ::AbstractRNG,
) = _CausalProposalState(proposal)

function BAT.mcmc_propose_transition(
    current_z::ArrayOfSimilarArrays,
    state::_CausalProposalState,
    ::AbstractVector,
)
    proposal = state.proposal
    handshake = proposal.handshake
    if proposal.role == :proposal
        take!(handshake.sibling_started)
        put!(handshake.failing_task, current_task())
        error("walker-isolation proposal failure")
    elseif proposal.role in (:target, :output)
        take!(handshake.sibling_started)
    elseif proposal.role == :sibling
        step = Threads.atomic_add!(handshake.sibling_steps, 1) + 1
        if step == 1
            put!(handshake.sibling_started, nothing)
            task = take!(handshake.failing_task)
            try
                wait(task)
            catch err
                err isa TaskFailedException || rethrow()
            end
        end
    end
    transition = [fill(0.25, length(z)) for z in current_z]
    return current_z .+ transition, zeros(length(current_z))
end
_fixed_algorithm(
    proposal;
    nwalkers = 1,
    proposal_tuning = NoMCMCProposalTuning(),
    adaptive_transform = NoAdaptiveTransform(),
    transform_tuning = NoMCMCTransformTuning(),
) = TransformedMCMC(
    proposal = proposal,
    proposal_tuning = proposal_tuning,
    pretransform = DoNotTransform(),
    adaptive_transform = adaptive_transform,
    transform_tuning = transform_tuning,
    tempering = BAT.NoMCMCTempering(),
    nwalkers = nwalkers,
    nonzero_weights = false,
)

_zero_initialized_chain(algorithm, target, chain_id, seed) = BAT.MCMCState(
    algorithm,
    target,
    chain_id,
    [zeros(2)],
    BATContext(rng = Philox4x(seed), ad = ForwardDiff),
)
function _set_logical_walker_ids!(state, walker_ids)
    chain_state = state.chain_state
    for samples in (
        chain_state.current.x,
        chain_state.current.z,
        chain_state.proposed.x,
        chain_state.proposed.z,
        chain_state.output,
    )
        for i in eachindex(samples)
            info = samples.info[i]
            samples.info[i] = BAT.MCMCSampleID(
                info.chainid,
                Int32(walker_ids[i]),
                info.chaincycle,
                info.stepno,
                info.proposalid,
                info.sampletype,
            )
        end
    end
    chain_state.walker_order .= sortperm(walker_ids)
    return state
end
function _walker_signatures(
    proposal,
    initial_positions,
    walker_ids;
    seed,
    nsteps = 1,
    chain_partition = false,
    proposal_tuning = NoMCMCProposalTuning(),
    adaptive_transform = NoAdaptiveTransform(),
    transform_tuning = NoMCMCTransformTuning(),
)
    algorithm = _fixed_algorithm(
        deepcopy(proposal);
        nwalkers = length(walker_ids),
        proposal_tuning = proposal_tuning,
        adaptive_transform = adaptive_transform,
        transform_tuning = transform_tuning,
    )
    rng = Philox4x(seed)
    if chain_partition
        rngpart = BAT.RNGPartition(rng, Base.OneTo(1))
        rng = AbstractRNG(rngpart, 1)
    end
    state = BAT.MCMCState(
        algorithm,
        batmeasure(MvNormal(zeros(2), I)),
        1,
        initial_positions,
        BATContext(rng = rng, ad = ForwardDiff),
    )
    _set_logical_walker_ids!(state, walker_ids)
    BAT.mcmc_tuning_init!!(state, nsteps)
    BAT.mcmc_tuning_reinit!!(state, nsteps)
    outputs = BAT._empty_chain_outputs(state)
    state = BAT.mcmc_iterate!!(
        outputs,
        state;
        max_nsteps = nsteps,
        nonzero_weights = false,
    )
    chain_state = state.chain_state
    paths = Dict(
        walker_ids[i] => (
            current_v = copy(chain_state.current.x.v[i]),
            current_logd = chain_state.current.x.logd[i],
            output_v = copy.(outputs[i].v),
            output_logd = copy(outputs[i].logd),
            output_weight = copy(outputs[i].weight),
        ) for i in eachindex(walker_ids)
    )
    ids = Dict(
        walker_ids[i] => (
            current = chain_state.current.x.info[i],
            output = copy(outputs[i].info),
        ) for i in eachindex(walker_ids)
    )
    diagnostics = _diagnostics_signature(state)
    tuning = let tuner = state.proposal_tuner_state
        if tuner isa BAT.HMCStepSizeTunerState
            proposal_state = BAT.get_active_proposal(chain_state.proposal)
            (
                proposal_state.step_size,
                tuner.m,
                tuner.log_mu,
                tuner.log_stepsize_bar,
                tuner.H_bar,
                tuner.run_nobs,
                tuner.run_accept_sum,
                tuner.run_accept_sqsum,
                tuner.run_ndivergent,
                tuner.run_skip,
                tuner.min_run_nobs,
            )
        elseif tuner isa BAT.MALAStepSizeTunerState
            proposal_state = BAT.get_active_proposal(chain_state.proposal)
            (
                proposal_state.τ,
                tuner.m,
                tuner.log_mu,
                tuner.log_stepsize_bar,
                tuner.H_bar,
                tuner.run_nobs,
                tuner.run_accept_sum,
                tuner.min_run_nobs,
            )
        elseif tuner isa BAT.AdaptiveMultiPropTunerState
            picking_rule = chain_state.proposal.picking_rule
            (
                chain_state.proposal.active_idx,
                copy(picking_rule.p),
                copy(tuner.accept_prob),
            )
        else
            nothing
        end
    end
    transform = let f = chain_state.f_transform
        f isa BAT.MulAdd ? (A = copy(f.A), b = copy(f.b)) : nothing
    end
    trafo_tuning = let tuner = state.trafo_tuner_state
        if tuner isa BAT.FisherTrafoTunerState
            (_moments_signature(tuner.acc_a), _moments_signature(tuner.acc_b))
        elseif tuner isa BAT.RAMTrafoTunerState
            (tuner.tuning, tuner.nsteps)
        else
            nothing
        end
    end
    return (; paths, ids, diagnostics, tuning, transform, trafo_tuning)
end
_sample_signature(samples) = (
    v = copy.(samples.v),
    logd = copy(samples.logd),
    weight = copy(samples.weight),
    info = copy(samples.info),
)

_output_signature(outputs) = [_sample_signature(output) for output in outputs]

_stats_signature(stats) = (
    mean = copy(stats.param_stats.mean),
    cov = copy(stats.param_stats.cov),
    maximum = copy(stats.param_stats.maximum),
    minimum = copy(stats.param_stats.minimum),
    mode = copy(stats.mode),
)

function _tuner_order_state(proposal, adaptive_transform, transform_tuning, values, walker_ids)
    algorithm = TransformedMCMC(
        proposal = proposal,
        proposal_tuning = NoMCMCProposalTuning(),
        pretransform = DoNotTransform(),
        adaptive_transform = adaptive_transform,
        transform_tuning = transform_tuning,
        tempering = BAT.NoMCMCTempering(),
        nwalkers = length(walker_ids),
        nonzero_weights = false,
    )
    state = BAT.MCMCState(
        algorithm,
        batmeasure(MvNormal(zeros(2), I)),
        1,
        values,
        BATContext(rng = Philox4x((0x0564, 40)), ad = ForwardDiff),
    )
    _set_logical_walker_ids!(state, walker_ids)
    BAT.mcmc_tuning_init!!(state, 8)
    BAT.mcmc_tuning_reinit!!(state, 8)
    return state
end

function _moments_signature(acc)
    lag1 = acc.lag1
    return (
        acc.n,
        copy(acc.mean_x),
        copy(acc.mean_g),
        copy(acc.M2_x),
        copy(acc.M2_g),
        (lag1.n1, lag1.filled, lag1.ptr, copy(lag1.prev), copy(lag1.cross1)),
    )
end

function _post_step_tuner_signature(
    proposal, adaptive_transform, transform_tuning, values, walker_ids,
)
    state = _tuner_order_state(
        proposal, adaptive_transform, transform_tuning, values, walker_ids,
    )
    chain_state = state.chain_state
    chain_state.accepted .= false
    step_info = BAT.MCMCStepInfo(
        ones(length(walker_ids)), nothing, nothing, nothing, nothing,
        BAT._logical_walker_order(chain_state),
    )
    BAT.mcmc_tune_trafo_post_step!!(
        chain_state.f_transform, state.trafo_tuner_state, chain_state,
        BAT.get_active_proposal(chain_state.proposal), chain_state.current,
        chain_state.proposed, step_info,
    )
    return _stats_signature(state.trafo_tuner_state.stats)
end

function _postinit_tuner_signature(values, walker_ids)
    state = _tuner_order_state(
        RandomWalk(), BAT.TriangularAffineTransform(init = BAT.UnitTransformInit()),
        AdaptiveAffineTuning(), values, walker_ids,
    )
    samples = [DensitySampleVector(
        v = [value], logd = [-sum(abs2, value) / 2], weight = [1.0],
    ) for value in values]
    BAT.mcmc_tuning_postinit!!(state, samples)
    return _stats_signature(state.trafo_tuner_state.stats)
end

function _diagnostics_signature(state)
    proposal = BAT.get_active_proposal(state.chain_state.proposal)
    proposal isa BAT.HMCProposalState || return nothing
    diag = proposal.diagnostics
    return (
        diag.n_transitions,
        diag.n_divergent,
        diag.n_maxdepth,
        diag.n_leapfrog,
        diag.sum_p_accept,
    )
end

function _chain_prefix_signature(state, outputs)
    chain_state = state.chain_state
    return (
        stepno = chain_state.stepno,
        current_x = _sample_signature(chain_state.current.x),
        current_z = _sample_signature(chain_state.current.z),
        output = _output_signature(outputs),
        diagnostics = _diagnostics_signature(state),
    )
end

function _serial_prefix(algorithm, chain_id, seed, nsteps)
    state = _zero_initialized_chain(
        algorithm, batmeasure(MvNormal(zeros(2), I)), chain_id, seed,
    )
    outputs = BAT._empty_chain_outputs(state)
    state = BAT.mcmc_iterate!!(
        outputs, state; max_nsteps = nsteps, nonzero_weights = false,
    )
    return _chain_prefix_signature(state, outputs)
end

function _causal_failure_case(source; collect_output = true)
    handshake = _FailureHandshake()
    failing_target = source == :target ?
        _CausalTarget(handshake, Threads.Atomic{Bool}(false)) : MvNormal(zeros(2), I)
    failing_algorithm = _fixed_algorithm(_CausalProposal(source, handshake))
    sibling_algorithm = _fixed_algorithm(_CausalProposal(:sibling, handshake))
    states = MCMCState[
        _zero_initialized_chain(failing_algorithm, batmeasure(failing_target), 1, (0x0564, 43)),
        _zero_initialized_chain(
            sibling_algorithm, batmeasure(MvNormal(zeros(2), I)), 2, (0x0564, 44),
        ),
    ]
    failing_target isa _CausalTarget && (failing_target.armed[] = true)
    sibling_output = BAT._empty_chain_outputs(states[2])
    outputs = if !collect_output
        nothing
    elseif source == :output
        T = eltype(sibling_output)
        AbstractVector{T}[_FailingOutputs{T}(handshake), sibling_output]
    else
        [BAT._empty_chain_outputs(states[1]), sibling_output]
    end

    err = _capture_error() do
        BAT.mcmc_iterate!!(outputs, states; max_nsteps = 200, nonzero_weights = false)
    end
    steps_at_return = handshake.sibling_steps[]
    yield()
    serial = _serial_prefix(
        _fixed_algorithm(_CausalProposal(:serial, handshake)), 2, (0x0564, 44), 1,
    )
    return (; err, states, outputs, sibling_output, serial, handshake, steps_at_return)
end

function _capture_error(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end


@testset "MCMC walker isolation" begin
    @testset "RandomWalk follows logical walkers, not storage order" begin
        proposal = RandomWalk(proposaldist = Normal())
        positions = [[0.0, 0.0], [4.0, 4.0]]
        reference = _walker_signatures(proposal, positions, [1, 2]; seed = (0x0564, 4))
        permuted = _walker_signatures(proposal, reverse(positions), [2, 1]; seed = (0x0564, 4))

        @test permuted.paths == reference.paths
        @test permuted.ids == reference.ids
    end

    @testset "walker streams fit the native chain RNG partition" begin
        err = _capture_error() do
            _walker_signatures(
                RandomWalk(proposaldist = Normal()),
                [[0.0, 0.0], [4.0, 4.0]],
                [1, 2];
                seed = (0x0564, 11),
                chain_partition = true,
            )
        end

        @test isnothing(err)
    end

    @testset "stochastic adaptive multi-proposal follows logical walkers" begin
        proposal = MCMCMultiProposal(
            proposals = BAT.MCMCProposal[
                RandomWalk(proposaldist = Normal()),
                RandomWalk(proposaldist = TDist(5)),
            ],
            picking_rule = Categorical([0.6, 0.4]),
        )
        rng = MersenneTwister(0x0565)
        nwalkers = 16
        positions = [randn(rng, 2) for _ in 1:nwalkers]
        permutation = randperm(rng, nwalkers)
        reference = _walker_signatures(
            proposal,
            positions,
            collect(1:nwalkers);
            seed = (0x0564, 24),
            nsteps = 8,
            proposal_tuning = AdaptiveMultiPropTuning(),
        )
        permuted = _walker_signatures(
            proposal,
            positions[permutation],
            permutation;
            seed = (0x0564, 24),
            nsteps = 8,
            proposal_tuning = AdaptiveMultiPropTuning(),
        )

        @test permuted.paths == reference.paths
        @test permuted.ids == reference.ids
        @test permuted.tuning == reference.tuning
        selected_proposals = reduce(vcat, [
            getproperty.(entry.output, :proposalid) for entry in values(reference.ids)
        ])
        @test length(unique(selected_proposals)) > 1
    end

    @testset "global proposals follow logical walkers" begin
        proposal = MCMCGlobalProposal(global_proposal = MvNormal(zeros(2), I))
        positions = [[0.0, 0.0], [4.0, 4.0]]
        reference = _walker_signatures(
            proposal, positions, [1, 2]; seed = (0x0564, 25), nsteps = 3,
        )
        permuted = _walker_signatures(
            proposal, reverse(positions), [2, 1]; seed = (0x0564, 25), nsteps = 3,
        )

        @test permuted.paths == reference.paths
        @test permuted.ids == reference.ids
    end

    @testset "acceptance draws are separate from transition draw counts" begin
        positions = [[-2.0, -2.0], [0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]
        walker_ids = collect(1:length(positions))
        reference = _walker_signatures(
            _FixedTransitionProposal(0),
            positions,
            walker_ids;
            seed = (0x0564, 26),
            nsteps = 5,
        )
        extra_draws = _walker_signatures(
            _FixedTransitionProposal(50),
            positions,
            walker_ids;
            seed = (0x0564, 26),
            nsteps = 5,
        )

        @test extra_draws.paths == reference.paths
        @test extra_draws.ids == reference.ids
    end

    @testset "next_cycle preserves noncanonical logical walker IDs" begin
        state = BAT.MCMCState(
            _fixed_algorithm(RandomWalk(); nwalkers = 2),
            batmeasure(MvNormal(zeros(2), I)),
            3,
            [[0.0, 0.0], [1.0, 1.0]],
            BATContext(rng = Philox4x((0x0564, 27))),
        )
        _set_logical_walker_ids!(state, [9, 3])
        BAT.next_cycle!(state)

        for (samples, sampletype) in (
            (state.chain_state.current.x, true),
            (state.chain_state.current.z, true),
            (state.chain_state.proposed.x, false),
            (state.chain_state.proposed.z, false),
        )
            @test getproperty.(samples.info, :walkerid) == Int32[9, 3]
            @test all(
                ==(state.chain_state.info.cycle),
                getproperty.(samples.info, :chaincycle),
            )
            @test getproperty.(samples.info, :stepno) == [0, 0]
            @test all(==(sampletype), getproperty.(samples.info, :sampletype))
        end
    end

    @testset "active HMC and MALA tuning follows logical walker order" begin
        rng = MersenneTwister(0x0564)
        nwalkers = 16
        positions = [2 .* randn(rng, 2) for _ in 1:nwalkers]
        permutation = randperm(rng, nwalkers)

        for (proposal, seed) in (
            (HamiltonianMC(step_size = 0.3, max_depth = 5), (0x0564, 20)),
            (MALAProposal(τ_base = 0.25), (0x0564, 21)),
        )
            reference = _walker_signatures(
                proposal,
                positions,
                collect(1:nwalkers);
                seed,
                nsteps = 4,
                proposal_tuning = BAT.StepSizeAdaptor(),
                adaptive_transform = BAT.DiagonalAffineTransform(
                    init = BAT.UnitTransformInit(),
                ),
                transform_tuning = FisherTransformTuning(),
            )
            permuted = _walker_signatures(
                proposal,
                positions[permutation],
                permutation;
                seed,
                nsteps = 4,
                proposal_tuning = BAT.StepSizeAdaptor(),
                adaptive_transform = BAT.DiagonalAffineTransform(
                    init = BAT.UnitTransformInit(),
                ),
                transform_tuning = FisherTransformTuning(),
            )

            @test permuted.paths == reference.paths
            @test permuted.ids == reference.ids
            @test permuted.diagnostics == reference.diagnostics
            @test permuted.tuning == reference.tuning
            @test permuted.transform == reference.transform
            @test permuted.trafo_tuning == reference.trafo_tuning
        end
    end

    @testset "order-sensitive transform tuners follow logical walkers" begin
        values = [[1.0e16, 1.0], [1.0, -2.0], [-1.0e16, 3.0], [4.0, -4.0]]
        permutation = [3, 1, 4, 2]
        ids = collect(eachindex(values))

        stan_tuning = BAT.StanLikeTuning(init_buffer = 0, term_buffer = 0, window_size = 8)
        reference = _post_step_tuner_signature(
            RandomWalk(), BAT.TriangularAffineTransform(init = BAT.UnitTransformInit()),
            stan_tuning, values, ids,
        )
        permuted = _post_step_tuner_signature(
            RandomWalk(), BAT.TriangularAffineTransform(init = BAT.UnitTransformInit()),
            stan_tuning, values[permutation], ids[permutation],
        )
        @test permuted == reference
        @test _postinit_tuner_signature(values[permutation], ids[permutation]) ==
            _postinit_tuner_signature(values, ids)
    end

    @testset "RAM transform tuning follows logical walker order" begin
        rng = MersenneTwister(0x0566)
        nwalkers = 16
        positions = [2 .* randn(rng, 2) for _ in 1:nwalkers]
        permutation = randperm(rng, nwalkers)
        kwargs = (
            seed = (0x0564, 30),
            nsteps = 4,
            adaptive_transform = BAT.TriangularAffineTransform(),
            transform_tuning = RAMTuning(),
        )
        proposal = RandomWalk(proposaldist = Normal())
        reference = _walker_signatures(
            proposal, positions, collect(1:nwalkers); kwargs...,
        )
        permuted = _walker_signatures(
            proposal, positions[permutation], permutation; kwargs...,
        )

        @test permuted.paths == reference.paths
        @test permuted.transform == reference.transform
        @test permuted.trafo_tuning == reference.trafo_tuning
    end

    @testset "cancellation flushes the exact completed serial prefix" begin
        for collect_output in (true, false)
            algorithm = _fixed_algorithm(RandomWalk())
            stepped = _zero_initialized_chain(
                algorithm, batmeasure(MvNormal(zeros(2), I)), 1, (0x0564, 22),
            )
            stepped = BAT.mcmc_step!!(stepped)
            cancelled_state = deepcopy(stepped)
            serial_state = deepcopy(stepped)
            cancelled_output = collect_output ? BAT._empty_chain_outputs(cancelled_state) : nothing
            serial_output = collect_output ? BAT._empty_chain_outputs(serial_state) : nothing

            cancelled_state = BAT.mcmc_iterate!!(
                cancelled_output,
                cancelled_state;
                max_nsteps = 10,
                nonzero_weights = false,
                _cancelled = Threads.Atomic{Bool}(true),
            )
            serial_state = BAT.mcmc_iterate!!(
                serial_output, serial_state; max_nsteps = 0, nonzero_weights = false,
            )

            collect_output && @test(
                _output_signature(cancelled_output) == _output_signature(serial_output)
            )
            @test cancelled_state.chain_state.current.x == serial_state.chain_state.current.x
            @test cancelled_state.chain_state.current.z == serial_state.chain_state.current.z
            @test cancelled_state.chain_state.output == serial_state.chain_state.output
        end
    end

    @testset "proposal-purpose stream indices are bounded and collision-free" begin
        max_proposals = BAT._MCMC_PROPOSALS_PER_PURPOSE
        transition_max = BAT._mcmc_rng_stream_idx(
            BAT._MCMC_PROPOSAL_TRANSITION_PURPOSE, max_proposals,
        )
        acceptance_min = BAT._mcmc_rng_stream_idx(BAT._MCMC_ACCEPTANCE_PURPOSE, 1)

        @test transition_max != acceptance_min
        @test_throws ArgumentError BAT._mcmc_rng_stream_idx(
            BAT._MCMC_PROPOSAL_TRANSITION_PURPOSE, 0,
        )
        @test_throws ArgumentError BAT._mcmc_rng_stream_idx(
            BAT._MCMC_PROPOSAL_TRANSITION_PURPOSE, max_proposals + 1,
        )
        @test_throws ArgumentError BAT._mcmc_rng_stream_idx(0, 1)
        @test_throws ArgumentError BAT._mcmc_rng_stream_idx(
            BAT._MCMC_N_RNG_PURPOSES + 1, 1,
        )

        target = batmeasure(MvNormal(zeros(2), I))
        context = BATContext(rng = Philox4x((0x0564, 28)))
        empty_proposal = MCMCMultiProposal(
            proposals = BAT.MCMCProposal[], picking_rule = Int[],
        )
        excessive_proposal = MCMCMultiProposal(
            proposals = fill(RandomWalk(), max_proposals + 1),
            picking_rule = ones(Int, max_proposals + 1),
        )
        @test_throws ArgumentError BAT.MCMCState(
            _fixed_algorithm(empty_proposal), target, 1, [zeros(2)], context,
        )
        @test_throws ArgumentError BAT.MCMCState(
            _fixed_algorithm(excessive_proposal), target, 1, [zeros(2)], context,
        )
    end

    @testset "causal $source failure bounds sibling work" for source in (:target, :proposal, :output)
        result = _causal_failure_case(source)
        err = result.err

        @test err isa CompositeException
        @test all(exception -> exception isa TaskFailedException, err.exceptions)
        @test all(!isempty(Base.current_exceptions(exception.task)) for exception in err.exceptions)
        @test occursin("walker-isolation $source failure", sprint(showerror, err))
        @test result.steps_at_return == 1
        @test result.handshake.sibling_steps[] == 1
        @test maximum(result.sibling_output[1].info.stepno; init = 0) == 1
        @test _output_signature(result.sibling_output) == result.serial.output
        @test _chain_prefix_signature(result.states[2], result.sibling_output) == result.serial
    end

    @testset "causal cancellation works without output collection" begin
        result = _causal_failure_case(:proposal; collect_output = false)

        @test result.err isa CompositeException
        @test result.steps_at_return == 1
        @test result.handshake.sibling_steps[] == 1
        @test result.states[2].chain_state.stepno == 1
        @test _sample_signature(result.states[2].chain_state.current.x) == result.serial.current_x
        @test _sample_signature(result.states[2].chain_state.current.z) == result.serial.current_z
    end

    @testset "MCMCRetryInit rerolls an unviable logical walker reproducibly" begin
        function run_retry_init()
            calls = Threads.Atomic{Int}(0)
            init_alg = MCMCRetryInit(
                max_init_tries = 2,
                nsteps_init = 10,
                initval_alg = _ScriptedInit(calls),
                strict = true,
            )
            algorithm = TransformedMCMC(
                proposal = _FixedTransitionProposal(0),
                proposal_tuning = NoMCMCProposalTuning(),
                pretransform = DoNotTransform(),
                adaptive_transform = NoAdaptiveTransform(),
                transform_tuning = NoMCMCTransformTuning(),
                tempering = BAT.NoMCMCTempering(),
                init = init_alg,
                nchains = 1,
                nwalkers = 2,
                nonzero_weights = true,
                sample_weighting = _ViabilityWeighting(),
            )
            result = BAT.mcmc_init!(
                algorithm,
                batmeasure(product_distribution(fill(Uniform(-2, 2), 2))),
                init_alg,
                (args...) -> nothing,
                BATContext(rng = Philox4x((0x0564, 29))),
            )
            state = only(result.mcmc_states)
            return (
                calls = calls[],
                walker_ids = getproperty.(state.chain_state.current.x.info, :walkerid),
                signature = _chain_prefix_signature(state, only(result.outputs)),
            )
        end

        reference = run_retry_init()
        replay = run_retry_init()

        @test reference.calls == 3
        @test reference.walker_ids == Int32[1, 2]
        @test !isempty(reference.signature.output[1].v)
        @test replay == reference
    end
end

end
