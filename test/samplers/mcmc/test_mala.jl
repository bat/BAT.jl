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

@testset "mala" begin
    rng = StableRNG(902114857)

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

    @testset "gradient cache scalar type" begin
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
        @test eltype(only(cache.grads_prop)) === T

        source = BigFloat[big"-1.0" + big"1e-30", 0]
        cache_big = BAT._MALAGradCache([zeros(BigFloat, 1)])
        cache_big.grads_curr = [view(source, 1:1)]
        @test only(only(cache_big.grads_curr)) == first(source)
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
