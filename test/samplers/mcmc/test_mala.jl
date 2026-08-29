# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, Statistics
using Distributions, ValueShapes, DensityInterface
using StableRNGs
using Random123
import ForwardDiff

using BAT: MALAProposal, StepSizeAdaptor, LowRankAffineTransform,
    _mala_innovation_dist, _mala_log_proposal_ratio, batmeasure

struct _CountingMALATarget{M} <: BAT.BATMeasure
    base::M
    calls::Base.RefValue{Int}
end

DensityInterface.logdensityof(m::_CountingMALATarget, x) =
    (m.calls[] += 1; logdensityof(m.base, x))
ValueShapes.varshape(m::_CountingMALATarget) = ValueShapes.varshape(m.base)

struct _ShiftedMALATarget{M,T<:Real} <: BAT.BATMeasure
    base::M
    shift::T
end

DensityInterface.logdensityof(m::_ShiftedMALATarget, x) =
    m.shift + logdensityof(m.base, x)
ValueShapes.varshape(m::_ShiftedMALATarget) = ValueShapes.varshape(m.base)

@testset "mala" begin
    rng = StableRNG(902114857)

    function make_fixed_mala_state(target, initial, τ_base, seed)
        algorithm = TransformedMCMC(
            proposal = MALAProposal(τ_base = τ_base),
            proposal_tuning = BAT.NoMCMCProposalTuning(),
            adaptive_transform = BAT.NoAdaptiveTransform(),
            pretransform = DoNotTransform(),
            nchains = 1,
            nwalkers = 1,
        )
        return BAT.MCMCState(
            algorithm,
            target,
            1,
            [initial],
            BATContext(
                precision = eltype(initial),
                rng = Philox4x((564, seed)),
                ad = ForwardDiff,
            ),
        )
    end

    @testset "innovation distribution" begin
        # The Langevin innovation acts at unit scale, unlike random-walk
        # proposal distributions (which carry the optimal-scaling factor):
        d = _mala_innovation_dist(Normal(), 5)
        @test length(d) == 5
        @test var(d) == fill(1.0, 5)
    end

    @testset "exact proposal log ratio" begin
        τ = 0.3
        n = 3
        Δ = randn(rng, n)
        g_x = randn(rng, n)
        g_y = randn(rng, n)

        # Gaussian innovation: must match the closed-form Gaussian MALA
        # Hastings correction with the τ/2 drift:
        pm = batmeasure(_mala_innovation_dist(Normal(), n))
        h = only(_mala_log_proposal_ratio(pm, τ, [Δ], [g_x], [g_y]))
        h_ref = (sum(abs2, Δ .- τ/2 .* g_x) - sum(abs2, .-Δ .- τ/2 .* g_y)) / (2τ)
        @test h ≈ h_ref

        # Non-Gaussian innovation: must match the exact log-density ratio
        # of the actual innovation distribution:
        pm_t = batmeasure(_mala_innovation_dist(TDist(4.0), n))
        ξ_fwd = (Δ .- τ/2 .* g_x) ./ sqrt(τ)
        ξ_rev = (.-Δ .- τ/2 .* g_y) ./ sqrt(τ)
        h_t = only(_mala_log_proposal_ratio(pm_t, τ, [Δ], [g_x], [g_y]))
        @test h_t ≈ logdensityof(pm_t, ξ_rev) - logdensityof(pm_t, ξ_fwd)
    end

    @testset "gradient cache scalar type and density authority" begin
        T = Float32
        target = batmeasure(MvNormal(zeros(T, 2), Diagonal(ones(T, 2))))
        algorithm = TransformedMCMC(
            proposal = MALAProposal(),
            adaptive_transform = BAT.DiagonalAffineTransform(),
            pretransform = DoNotTransform(),
            nchains = 1,
            nwalkers = 1,
        )
        state = BAT.MCMCState(
            algorithm,
            target,
            1,
            [T[0.25, -0.5]],
            BATContext(precision = T, rng = Philox4x((564, 9)), ad = ForwardDiff),
        )
        BAT.mcmc_tuning_init!!(state, 100)
        BAT.mcmc_tuning_reinit!!(state, 100)
        state = BAT.mcmc_iterate!!(nothing, state; max_nsteps = 1, nonzero_weights = false)
        cache = BAT.get_active_proposal(state.chain_state.proposal).grad_cache
        @test eltype(only(cache.grads_curr)) === T

        target_wide = _ShiftedMALATarget(
            batmeasure(MvNormal(zeros(T, 2), Diagonal(ones(T, 2)))), 1.0e8
        )
        state_wide = make_fixed_mala_state(
            target_wide, Float32[0.1, -0.2], 0.2, 1311
        )
        cache_wide = BAT.get_active_proposal(state_wide.chain_state.proposal).grad_cache
        @test eltype(state_wide.chain_state.current.z.logd) === Float64
        @test eltype(cache_wide.grads_curr) === Vector{Float32}
        @test isempty(cache_wide.grads_curr)

        n_rng_streams = BAT._MCMC_N_RNG_PURPOSES * BAT._MCMC_PROPOSALS_PER_PURPOSE
        oracle = deepcopy(state_wide)
        oracle_chain_state = oracle.chain_state
        oracle_step_rngpart = BAT.RNGPartition(
            BAT.get_rng(oracle_chain_state.context), Base.OneTo(n_rng_streams),
        )
        proposal_rngpart = BAT._mcmc_walker_rngpart(
            oracle_step_rngpart, BAT._MCMC_PROPOSAL_TRANSITION_PURPOSE, 1,
        )
        for i in eachindex(
            oracle_chain_state.walker_genctxs, oracle_chain_state.current.x.info,
        )
            genctx = oracle_chain_state.walker_genctxs[i]
            walkerid = oracle_chain_state.current.x.info[i].walkerid
            BAT.set_rng!(BAT.get_rng(genctx), proposal_rngpart, walkerid)
        end
        oracle_proposal = BAT.get_active_proposal(oracle_chain_state.proposal)
        z_proposed, hastings = BAT.mcmc_propose_transition(
            oracle_chain_state.current.z.v,
            oracle_proposal,
            oracle_chain_state.walker_genctxs,
        )
        x_proposed, ladj = BAT._transform_with_ladj(
            oracle_chain_state.f_transform, z_proposed,
        )
        logd_x_proposed = BAT.checked_logdensityof.(target_wide, x_proposed)
        p_accept = clamp.(exp.(
            logd_x_proposed .+ ladj .-
            oracle_chain_state.current.z.logd .+ hastings
        ), 0, 1)

        proposal = BAT.get_active_proposal(state_wide.chain_state.proposal)
        step_rngpart = BAT.RNGPartition(
            BAT.get_rng(state_wide.chain_state.context), Base.OneTo(n_rng_streams),
        )
        chain_state, _, step_info = BAT.mcmc_propose!!(
            state_wide.chain_state,
            proposal,
            step_rngpart,
            1,
            state_wide.chain_state.walker_order,
        )

        @test chain_state.proposed.z.v ≈ z_proposed
        @test chain_state.proposed.x.logd == logd_x_proposed
        @test eltype(chain_state.proposed.x.logd) === Float64
        @test step_info.p_accept == p_accept
        selected_z = only(chain_state.accepted) ? only(z_proposed) :
            only(chain_state.current.z.v)
        @test only(step_info.z_grads) == Float32.(-selected_z)
    end

    @testset "accepted gradient reuse" begin
        for (τ_base, initial, seed, expected_accept) in (
            (0.2, [0.1, -0.2], 1304, true),
            (20.0, [2.0, -1.5], 1300, false),
        )
            calls = Ref(0)
            target = _CountingMALATarget(batmeasure(MvNormal(zeros(2), I)), calls)
            state = make_fixed_mala_state(target, initial, τ_base, seed)

            @test calls[] == 1
            cache = BAT.get_active_proposal(state.chain_state.proposal).grad_cache
            @test isempty(cache.grads_curr)
            @test isempty(cache.grads_prop)

            calls[] = 0
            state = BAT.mcmc_step!!(state)
            @test calls[] == 3
            @test only(state.chain_state.accepted) == expected_accept

            @test only(cache.grads_curr) == -only(state.chain_state.current.z.v)
            @test isempty(cache.grads_prop)

            calls[] = 0
            state = BAT.mcmc_step!!(state)
            @test calls[] == 2
            @test only(state.chain_state.current.x.logd) ==
                logdensityof(target.base, only(state.chain_state.current.x.v))
        end

        # Transform commits rebind the target-gradient and invalidate
        # gradients expressed in the old geometry:
        target = batmeasure(MvNormal(zeros(2), I))
        state = make_fixed_mala_state(target, [0.1, -0.2], 0.2, 1312)
        proposal = BAT.get_active_proposal(state.chain_state.proposal)
        BAT.mcmc_propose_transition(
            state.chain_state.current.z.v,
            proposal,
            state.chain_state.walker_genctxs,
        )
        @test !isempty(proposal.grad_cache.grads_curr)
        @test !isempty(proposal.grad_cache.grads_prop)
        proposal = BAT.set_proposal_transform!!(proposal, state.chain_state)
        @test isempty(proposal.grad_cache.grads_curr)
        @test isempty(proposal.grad_cache.grads_prop)
    end

    @testset "cache selection atomicity" begin
        target = batmeasure(MvNormal(zeros(2), I))
        state = make_fixed_mala_state(target, [0.1, -0.2], 0.2, 1307)
        proposal = BAT.get_active_proposal(state.chain_state.proposal)
        BAT.mcmc_propose_transition(
            state.chain_state.current.z.v,
            proposal,
            state.chain_state.walker_genctxs,
        )

        cache = proposal.grad_cache
        current_storage = cache.grads_curr
        proposed_gradient = only(cache.grads_prop)
        @test BAT._selected_z_grads(proposal, [true]) == [proposed_gradient]
        @test cache.grads_curr === current_storage
        @test only(cache.grads_curr) === proposed_gradient

        BAT._invalidate_mala_cache!!(proposal)
        BAT.mcmc_propose_transition(
            state.chain_state.current.z.v,
            proposal,
            state.chain_state.walker_genctxs,
        )
        push!(cache.grads_prop, only(cache.grads_prop))
        current_storage = cache.grads_curr
        proposed_storage = cache.grads_prop
        current_before = copy(current_storage)
        proposed_before = copy(proposed_storage)
        @test isnothing(BAT._selected_z_grads(proposal, [true]))
        @test cache.grads_curr === current_storage
        @test cache.grads_prop === proposed_storage
        @test cache.grads_curr == current_before
        @test cache.grads_prop == proposed_before
    end

    @testset "multi-proposal cache coherence" begin
        function make_multi_state(proposals, picking_rule, seed)
            target = _CountingMALATarget(unshaped(batmeasure(Normal())), Ref(0))
            algorithm = TransformedMCMC(
                proposal = MCMCMultiProposal(; proposals, picking_rule),
                proposal_tuning = MultiProposalTuning(BAT.MCMCProposalTuning[
                    BAT.NoMCMCProposalTuning() for _ in proposals
                ]),
                adaptive_transform = BAT.NoAdaptiveTransform(),
                pretransform = DoNotTransform(),
                nchains = 1,
                nwalkers = 1,
            )
            state = BAT.MCMCState(
                algorithm,
                target,
                1,
                [[0.7]],
                BATContext(rng = Philox4x((564, seed)), ad = ForwardDiff),
            )
            return state, target.calls
        end

        state, calls = make_multi_state(
            fill(MALAProposal(τ_base = 0.2), 4), [1, 1, 1, 1], 1308
        )
        @test calls[] == 1
        @test all(isempty(p.grad_cache.grads_curr) &&
            isempty(p.grad_cache.grads_prop) for
            p in state.chain_state.proposal.proposal_states)

        for proposal in state.chain_state.proposal.proposal_states
            BAT.mcmc_propose_transition(
                state.chain_state.current.z.v,
                proposal,
                state.chain_state.walker_genctxs,
            )
        end
        BAT._invalidate_mala_cache!!(state.chain_state.proposal)
        @test all(isempty(p.grad_cache.grads_curr) &&
            isempty(p.grad_cache.grads_prop) for
            p in state.chain_state.proposal.proposal_states)

        function clear_component_cache!(state, idx)
            cache = state.chain_state.proposal.proposal_states[idx].grad_cache
            empty!(cache.grads_curr)
            empty!(cache.grads_prop)
        end

        function test_matches_fresh_cache(candidate, oracle, idx)
            @test candidate.chain_state.accepted == oracle.chain_state.accepted
            @test candidate.chain_state.current.z.v == oracle.chain_state.current.z.v
            @test candidate.chain_state.current.z.logd == oracle.chain_state.current.z.logd
            candidate_cache =
                candidate.chain_state.proposal.proposal_states[idx].grad_cache
            oracle_cache = oracle.chain_state.proposal.proposal_states[idx].grad_cache
            @test candidate_cache.grads_curr == oracle_cache.grads_curr
            @test only(candidate_cache.grads_curr) ==
                -only(candidate.chain_state.current.z.v)
            @test isempty(candidate_cache.grads_prop)
        end

        mixed_cases = (
            ([1, 1], 1300, 0.05, true),
            ([1, 1], 1, 20.0, false),
        )
        for (picking_rule, seed, rw_scale, expected_rw_accept) in mixed_cases
            proposals = BAT.MCMCProposal[
                MALAProposal(τ_base = 0.2),
                RandomWalk(proposaldist = Normal(0, rw_scale)),
            ]
            state, calls = make_multi_state(proposals, picking_rule, seed)
            state = BAT.mcmc_step!!(state)
            @test state.chain_state.proposal.active_idx == 1
            z_before_rw = copy(state.chain_state.current.z.v)

            state = BAT.mcmc_step!!(state)
            @test state.chain_state.proposal.active_idx == 2
            @test only(state.chain_state.accepted) == expected_rw_accept
            @test (state.chain_state.current.z.v != z_before_rw) == expected_rw_accept

            candidate, oracle = deepcopy(state), deepcopy(state)
            clear_component_cache!(oracle, 1)
            calls[] = 0
            candidate = BAT.mcmc_step!!(candidate)
            candidate_calls = calls[]
            calls[] = 0
            oracle = BAT.mcmc_step!!(oracle)
            oracle_calls = calls[]
            @test candidate.chain_state.proposal.active_idx == 1
            @test oracle.chain_state.proposal.active_idx == 1
            @test candidate_calls == 2
            @test oracle_calls == 2
            test_matches_fresh_cache(candidate, oracle, 1)
        end

        proposals = BAT.MCMCProposal[
            MALAProposal(τ_base = 0.2), MALAProposal(τ_base = 0.3),
        ]
        state, calls = make_multi_state(
            proposals, Categorical([0.5, 0.5]), 7
        )
        state = BAT.mcmc_step!!(state)
        @test state.chain_state.proposal.active_idx == 1
        @test only(state.chain_state.accepted)
        @test only(state.chain_state.current.z.v) != [0.7]

        candidate, oracle = deepcopy(state), deepcopy(state)
        clear_component_cache!(oracle, 2)
        calls[] = 0
        candidate = BAT.mcmc_step!!(candidate)
        candidate_calls = calls[]
        calls[] = 0
        oracle = BAT.mcmc_step!!(oracle)
        oracle_calls = calls[]
        @test candidate.chain_state.proposal.active_idx == 2
        @test oracle.chain_state.proposal.active_idx == 2
        @test candidate_calls == 2
        @test oracle_calls == 2
        test_matches_fresh_cache(candidate, oracle, 2)
    end

    @testset "transformed gradient alignment" begin
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
        cache = BAT.get_active_proposal(state.chain_state.proposal).grad_cache
        gradient = only(cache.grads_curr)
        x = only(state.chain_state.current.x.v)
        z = only(state.chain_state.current.z.v)
        @test x == f(z)
        @test only(state.chain_state.current.x.logd) == logdensityof(target, x)
        @test gradient == f.A' * (-x)
    end

    @testset "fixed-kernel stationary law" begin
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
        draws = Matrix{Float64}(undef, 2, 20_000)
        for i in 1:21_000
            state = BAT.mcmc_step!!(state)
            i > 1_000 && (draws[:, i - 1_000] = only(state.chain_state.current.x.v))
        end
        @test vec(mean(draws; dims = 2)) ≈ μ atol = 0.04
        @test cov(draws; dims = 2) ≈ Σ atol = 0.07
    end

    @testset "startup domain validation" begin
        target = batmeasure(MvNormal(zeros(2), I))
        context = BATContext(rng = Philox4x((564, 10)), ad = ForwardDiff)

        function make_state(
            proposal, tuning = StepSizeAdaptor(); target_measure = target,
        )
            algorithm = TransformedMCMC(
                proposal = proposal,
                proposal_tuning = tuning,
                pretransform = DoNotTransform(),
                nchains = 1,
                nwalkers = 1,
            )
            return BAT.MCMCState(
                algorithm, target_measure, 1, [zeros(2)], deepcopy(context),
            )
        end

        for τ_base in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError make_state(MALAProposal(τ_base = τ_base))
        end
        for target_acceptance in (-0.1, 0.0, 1.0, 1.1, Inf, NaN)
            @test_throws ArgumentError make_state(
                MALAProposal(target_acceptance = target_acceptance),
            )
        end
        for target_acceptance_int in (
            (-0.1, 0.5), (0.5, 1.1), (0.5, 0.5), (0.7, 0.6),
            (NaN, 0.5), (0.5, NaN),
        )
            for proposal in (
                MALAProposal(target_acceptance_int = target_acceptance_int),
                HamiltonianMC(
                    target_acceptance_int = target_acceptance_int,
                    step_size = 0.1,
                ),
            )
                @test_throws ArgumentError make_state(proposal)
            end
        end

        startup_sentinel = ErrorException("startup validation reached target preparation")
        startup_calls = Ref(0)
        sentinel_target = batmeasure(PosteriorMeasure(
            logfuncdensity() do _
                startup_calls[] += 1
                throw(startup_sentinel)
            end,
            MvNormal(zeros(2), I),
        ))
        for target_acceptance_int in ((), (0.5,), (0.4, 0.6, 0.8))
            for proposal in (
                MALAProposal(target_acceptance_int = target_acceptance_int),
                HamiltonianMC(
                    target_acceptance_int = target_acceptance_int,
                    step_size = 0.1,
                ),
            )
                @test_throws ArgumentError make_state(
                    proposal; target_measure = sentinel_target,
                )
                @test iszero(startup_calls[])
            end
        end
        for proposal in (MALAProposal(), HamiltonianMC(step_size = 0.1))
            calls_before = startup_calls[]
            err = try
                make_state(proposal; target_measure = sentinel_target)
                nothing
            catch err
                err
            end
            @test !isnothing(err)
            @test startup_calls[] == calls_before + 1
        end

        invalid_tunings = (
            StepSizeAdaptor(gamma = 0.0),
            StepSizeAdaptor(gamma = -1.0),
            StepSizeAdaptor(gamma = Inf),
            StepSizeAdaptor(gamma = NaN),
            StepSizeAdaptor(t0 = -1.0),
            StepSizeAdaptor(t0 = Inf),
            StepSizeAdaptor(t0 = NaN),
            StepSizeAdaptor(kappa = 0.5),
            StepSizeAdaptor(kappa = 0.0),
            StepSizeAdaptor(kappa = 1.1),
            StepSizeAdaptor(kappa = Inf),
            StepSizeAdaptor(kappa = NaN),
        )
        for tuning in invalid_tunings
            @test_throws ArgumentError make_state(MALAProposal(), tuning)
            @test_throws ArgumentError make_state(
                HamiltonianMC(step_size = 0.1), tuning,
            )
        end

        valid_proposals = (
            MALAProposal(τ_base = floatmin(Float64)),
            MALAProposal(τ_base = floatmax(Float64)),
            MALAProposal(target_acceptance = nextfloat(0.0)),
            MALAProposal(target_acceptance = prevfloat(1.0)),
            MALAProposal(target_acceptance_int = (0.0, nextfloat(0.0))),
            MALAProposal(target_acceptance_int = (prevfloat(1.0), 1.0)),
        )
        for proposal in valid_proposals
            @test make_state(proposal) isa BAT.MCMCState
        end

        valid_tunings = (
            StepSizeAdaptor(gamma = floatmin(Float64)),
            StepSizeAdaptor(t0 = 0.0),
            StepSizeAdaptor(kappa = nextfloat(0.5)),
            StepSizeAdaptor(kappa = 1.0),
        )
        for tuning in valid_tunings
            @test make_state(MALAProposal(), tuning) isa BAT.MCMCState
            @test make_state(HamiltonianMC(step_size = 0.1), tuning) isa BAT.MCMCState
        end

        boundary_tuning = StepSizeAdaptor(gamma = floatmin(Float64), t0 = 0.0, kappa = 1.0)
        for initial_proposal in (MALAProposal(), HamiltonianMC(step_size = 0.1))
            for acceptance in (0.1, initial_proposal.target_acceptance, 0.95)
                state = make_state(initial_proposal, boundary_tuning)
                proposal = BAT.get_active_proposal(state.chain_state.proposal)
                tuner = state.proposal_tuner_state
                tuner_before = deepcopy(tuner)
                scale_before = proposal isa BAT.MALAProposalState ?
                    proposal.τ : proposal.step_size
                step_info = BAT.MCMCStepInfo([acceptance])

                if acceptance == initial_proposal.target_acceptance
                    proposal, tuner, chain_state = BAT.mcmc_tune_proposal_post_step!!(
                        proposal, tuner, state.chain_state, step_info,
                    )
                    scale = proposal isa BAT.MALAProposalState ? proposal.τ : proposal.step_size
                    @test isfinite(scale) && scale > 0
                    proposal, _, _ = BAT.mcmc_proposal_tuning_finalize!!(
                        proposal, tuner, chain_state,
                    )
                    scale = proposal isa BAT.MALAProposalState ? proposal.τ : proposal.step_size
                    @test isfinite(scale) && scale > 0
                else
                    @test_throws ArgumentError BAT.mcmc_tune_proposal_post_step!!(
                        proposal, tuner, state.chain_state, step_info,
                    )
                    @test all(fieldnames(typeof(tuner))) do name
                        getproperty(tuner, name) == getproperty(tuner_before, name)
                    end
                    scale = proposal isa BAT.MALAProposalState ? proposal.τ : proposal.step_size
                    @test scale == scale_before
                    proposal, _, _ = BAT.mcmc_proposal_tuning_finalize!!(
                        proposal, tuner, state.chain_state,
                    )
                    scale = proposal isa BAT.MALAProposalState ? proposal.τ : proposal.step_size
                    @test isfinite(scale) && scale > 0
                end
            end

            for invalid_log_scale in (-Inf, Inf)
                state = make_state(initial_proposal)
                proposal = BAT.get_active_proposal(state.chain_state.proposal)
                tuner = state.proposal_tuner_state
                tuner.m = 1
                tuner.log_stepsize_bar = invalid_log_scale
                @test_throws ArgumentError BAT.mcmc_proposal_tuning_finalize!!(
                    proposal, tuner, state.chain_state,
                )
            end
        end
    end

    @testset "sampling correctness" begin
        Σ = [1.0 0.6; 0.6 2.0]
        objective = MvNormal([1.0, -1.0], Σ)

        # Default MALA (Gaussian innovation, Fisher transform tuning):
        smplres = BAT.sample_and_verify(
            batmeasure(objective),
            TransformedMCMC(proposal = MALAProposal(), pretransform = DoNotTransform(), nsteps = 3 * 10^4),
            objective,
            BATContext(rng = Philox4x((564, 33)), ad = ForwardDiff),
            max_retries = 0,
        )
        @test smplres.verified
        @test smplres.n_retries == 0

        # The Fisher tuner sees coherent position/score pairs also under
        # MALA rejections, so the learned geometry matches the target:
        cs = BAT.samplegenof(smplres.evaluated).chain_states[1]
        f = cs.f_transform
        G_learned = Matrix(f.A * f.A')
        @test opnorm(G_learned - Σ) / opnorm(Σ) < 0.5

        # The step-scale adaptor steers the acceptance rate into the
        # target region (for MALA the state-movement rate is the
        # acceptance rate):
        @test 0.4 < BAT.eff_acceptance_ratio(cs) < 0.75

        # A heavy-tailed innovation is a valid generalized Langevin-MH
        # proposal now that the exact proposal densities are used:
        smplres_t = BAT.sample_and_verify(
            batmeasure(objective),
            TransformedMCMC(proposal = MALAProposal(proposaldist = TDist(4.0)), pretransform = DoNotTransform(), nsteps = 3 * 10^4),
            objective,
            BATContext(rng = Philox4x((564, 34)), ad = ForwardDiff),
            max_retries = 0,
        )
        @test smplres_t.verified
        @test smplres_t.n_retries == 0

        # Operator-valued low-rank transforms work with MALA: the gradient
        # uses the analytic affine pullback, so AD never sees the operator:
        u = normalize(fill(1.0, 3))
        Σ_lr = Matrix(Symmetric(Diagonal([1.0, 2.0, 0.5]) + 6.0 * u * u'))
        objective_lr = MvNormal(zeros(3), Σ_lr)
        smplres_lr = BAT.sample_and_verify(
            batmeasure(objective_lr),
            TransformedMCMC(
                proposal = MALAProposal(),
                adaptive_transform = LowRankAffineTransform(),
                pretransform = DoNotTransform(),
                nsteps = 3 * 10^4
            ),
            objective_lr,
            BATContext(rng = Philox4x((564, 35)), ad = ForwardDiff),
            max_retries = 0,
        )
        @test smplres_lr.verified
        @test smplres_lr.n_retries == 0
    end

    @testset "step scale adaptation" begin
        # Dual averaging moves τ against a persistent acceptance
        # imbalance:
        tuner_lo = BAT.MALAStepSizeTunerState(StepSizeAdaptor(), 0, log(10 * 0.5), 0.0, 0.0, 0, 0.0, 50)
        τ = 0.5
        for _ in 1:200
            τ = BAT._dual_averaging_step!(tuner_lo, 0.574, 0.1)
        end
        @test τ < 0.5

        tuner_hi = BAT.MALAStepSizeTunerState(StepSizeAdaptor(), 0, log(10 * 0.5), 0.0, 0.0, 0, 0.0, 50)
        τ = 0.5
        for _ in 1:200
            τ = BAT._dual_averaging_step!(tuner_hi, 0.574, 0.95)
        end
        @test τ > 0.5
    end

    @testset "provisional low-rank validation" begin
        d = 16
        u = normalize(ones(d))
        objective = MvNormal(
            zeros(d),
            Symmetric(Matrix{Float64}(I, d, d) + 16.0 * u * u'),
        )
        alg = TransformedMCMC(
            proposal = MALAProposal(),
            adaptive_transform = LowRankAffineTransform(
                init = BAT.UnitTransformInit(),
                cutoff = 1.5,
                max_rank = 1,
            ),
            pretransform = DoNotTransform(),
            nchains = 1,
            nwalkers = 1,
            convergence = AssumeConvergence(),
            nonzero_weights = false,
        )

        function run_decision(loss_sign, offdiag_sign, seed)
            initial = [rand(StableRNG(seed), objective)]
            state = BAT.MCMCState(
                alg,
                batmeasure(objective),
                1,
                initial,
                BATContext(
                    rng = Philox4x((seed + 1, seed + 2)),
                    ad = ForwardDiff,
                ),
            )
            BAT.mcmc_tuning_init!!(state, 1000)
            BAT.next_cycle!(state)
            BAT.mcmc_tuning_reinit!!(state, 1000)
            campaign = state.trafo_tuner_state.campaign
            @test campaign.fit_start == 201
            @test campaign.guard_steps == 64
            @test campaign.validation_steps == 512

            fit_end = campaign.fit_start + campaign.fit_steps - 1
            decision = fit_end + campaign.guard_steps + campaign.validation_steps
            provisional_f = nothing
            for step in 1:(decision - 1)
                f_before = state.chain_state.f_transform
                state = BAT.mcmc_step!!(state)
                if step == fit_end
                    @test !isnothing(campaign.candidate)
                    @test state.chain_state.f_transform !== f_before
                    provisional_f = state.chain_state.f_transform
                end
            end
            campaign.validation_loss .= loss_sign .* reshape(
                1e6 .+ 0.1 .* (-1.0) .^ (1:campaign.validation_steps),
                1,
                :,
            )
            if hasproperty(campaign, :validation_offdiag_loss)
                campaign.validation_offdiag_loss .= offdiag_sign .* reshape(
                    1e6 .+ 0.1 .* (-1.0) .^ (1:campaign.validation_steps),
                    1,
                    :,
                )
            end
            state = BAT.mcmc_step!!(state)
            return state, provisional_f
        end

        kept, provisional_f = run_decision(1.0, 1.0, 826_494_001)
        @test kept.trafo_tuner_state.campaign.admitted
        @test kept.chain_state.f_transform === provisional_f
        @test kept.proposal_tuner_state.min_run_nobs == 40

        rolled_back, provisional_f = run_decision(-1.0, 1.0, 826_494_002)
        @test !rolled_back.trafo_tuner_state.campaign.admitted
        @test rolled_back.chain_state.f_transform !== provisional_f
        G_rollback = Matrix(
            rolled_back.chain_state.f_transform.A *
            rolled_back.chain_state.f_transform.A',
        )
        @test G_rollback ≈ Diagonal(diag(G_rollback))
        @test rolled_back.proposal_tuner_state.min_run_nobs == 40

        offdiag_rejected, _ = run_decision(1.0, -1.0, 826_494_003)
        @test !offdiag_rejected.trafo_tuner_state.campaign.admitted
    end

    @testset "low-rank campaign lifecycle" begin
        objective = product_distribution(fill(TDist(3), 16))
        alg = TransformedMCMC(
            proposal = MALAProposal(),
            adaptive_transform = LowRankAffineTransform(),
            pretransform = DoNotTransform(),
            nchains = 1,
            nwalkers = 1,
            nsteps = 20,
            init = MCMCChainPoolInit(nsteps_init = 10),
            burnin = MCMCMultiCycleBurnin(
                nsteps_per_cycle = 1000,
                max_ncycles = 1,
                nsteps_final = 0,
            ),
            convergence = AssumeConvergence(),
        )
        state = BAT.MCMCState(
            alg,
            batmeasure(objective),
            1,
            [zeros(16)],
            BATContext(rng = Philox4x((42, 43)), ad = ForwardDiff),
        )
        BAT.mcmc_tuning_init!!(state, 1000)
        BAT.next_cycle!(state)
        BAT.mcmc_tuning_reinit!!(state, 1000)
        for _ in 1:1000
            state = BAT.mcmc_step!!(state)
        end
        campaign = state.trafo_tuner_state.campaign
        @test campaign.phase == BAT._LRFrozen
        @test campaign.attempted
        @test !BAT.transform_tuning_pauses_proposal(state.trafo_tuner_state)
    end
end
