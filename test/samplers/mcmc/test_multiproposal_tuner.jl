# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, Random123, ValueShapes
import ForwardDiff

struct _CountingProposalState <: BAT.MCMCProposalState
    updates::Int
end

struct _CountingTunerState <: BAT.MCMCProposalTunerState
    updates::Int
end

struct _CustomQualityProposalState <: BAT.MCMCProposalState end

BAT.get_proposal_tuning_quality(
    ::_CustomQualityProposalState,
    ::BAT.MCMCChainState,
    ::Float64,
) = 0.25

function _count_tuning_update(proposal, tuner, chain_state)
    chain_state_new = deepcopy(chain_state)
    chain_state_new.stepno += 1
    return typeof(proposal)(proposal.updates + 1), typeof(tuner)(tuner.updates + 1), chain_state_new
end

BAT.mcmc_proposal_tuning_finalize!!(
    proposal::_CountingProposalState,
    tuner::_CountingTunerState,
    chain_state::BAT.MCMCChainState,
) = _count_tuning_update(proposal, tuner, chain_state)

BAT.mcmc_tune_proposal_post_cycle!!(
    proposal::_CountingProposalState,
    tuner::_CountingTunerState,
    chain_state::BAT.MCMCChainState,
    ::AbstractVector{<:DensitySampleVector},
) = _count_tuning_update(proposal, tuner, chain_state)

BAT.mcmc_tune_proposal_post_step!!(
    proposal::_CountingProposalState,
    tuner::_CountingTunerState,
    chain_state::BAT.MCMCChainState,
    ::BAT.MCMCStepInfo,
) = _count_tuning_update(proposal, tuner, chain_state)

@testset "multi-proposal tuning" begin
    target = unshaped(batmeasure(Normal()))

    function make_state(
        proposal;
        proposal_tuning = nothing,
        context = BATContext(rng = Philox4x((87234, 1)), ad = ForwardDiff),
    )
        tuning_kw = isnothing(proposal_tuning) ? (;) : (; proposal_tuning)
        algorithm = TransformedMCMC(;
            proposal,
            pretransform = DoNotTransform(),
            adaptive_transform = BAT.NoAdaptiveTransform(),
            nwalkers = 1,
            tuning_kw...,
        )
        return BAT.MCMCState(algorithm, target, 1, [zeros(1)], context)
    end

    function construction_error(proposal; proposal_tuning = nothing, seed = 2)
        context = BATContext(rng = Philox4x((87234, seed)), ad = ForwardDiff)
        rng_before = deepcopy(BAT.get_rng(context))
        err = try
            make_state(proposal; proposal_tuning, context)
            nothing
        catch err
            err
        end
        rng_after = deepcopy(BAT.get_rng(context))
        return err, rand(rng_before, UInt64, 4) == rand(rng_after, UInt64, 4)
    end

    multiprop(proposals, rule) = MCMCMultiProposal(BAT.MCMCProposal[proposals...], rule)
    multituning(tunings...) = MultiProposalTuning(BAT.MCMCProposalTuning[tunings...])
    post_cycle(state) = BAT.mcmc_tune_proposal_post_cycle!!(
        state.chain_state.proposal,
        state.proposal_tuner_state,
        state.chain_state,
        DensitySampleVector[],
    )[1]

    @testset "configuration matrix" begin
        random_walks = BAT.MCMCProposal[RandomWalk(), RandomWalk()]
        fixed_tuning = multituning(NoMCMCProposalTuning(), NoMCMCProposalTuning())

        @test make_state(MCMCMultiProposal(random_walks, [1, 1]);
            proposal_tuning = fixed_tuning) isa BAT.MCMCState
        @test make_state(MCMCMultiProposal(random_walks, Categorical([0.5, 0.5]));
            proposal_tuning = fixed_tuning) isa BAT.MCMCState
        @test make_state(MCMCMultiProposal(random_walks, Categorical([0.5, 0.5]));
            proposal_tuning = AdaptiveMultiPropTuning()) isa BAT.MCMCState
        @test make_state(multiprop([MALAProposal(), HamiltonianMC()], [1, 1]);
            proposal_tuning = multituning(
                BAT.StepSizeAdaptor(), BAT.StepSizeAdaptor(),
            )) isa BAT.MCMCState
        @test make_state(HamiltonianMC();
            proposal_tuning = NoMCMCProposalTuning()) isa BAT.MCMCState
        @test make_state(multiprop([RandomWalk()], [1]);
            proposal_tuning = NoMCMCProposalTuning()) isa BAT.MCMCState
        narrow_schedule = make_state(MCMCMultiProposal(random_walks, Int8[100, 100]);
            proposal_tuning = fixed_tuning)
        for _ in 1:101
            narrow_schedule = BAT.mcmc_step!!(narrow_schedule)
        end
        @test narrow_schedule.chain_state.nattempts == [100, 1]

        zero_weight_schedules = [
            ([0, 2], [2, 2, 2, 2], [0, 4]),
            ([1, 0, 1], [1, 3, 1, 3], [2, 0, 2]),
            ([2, 0], [1, 1, 1, 1], [4, 0]),
        ]
        for (rule, expected_sequence, expected_attempts) in zero_weight_schedules
            proposals = BAT.MCMCProposal[RandomWalk() for _ in rule]
            tunings = BAT.MCMCProposalTuning[NoMCMCProposalTuning() for _ in rule]
            state = make_state(MCMCMultiProposal(proposals, rule);
                proposal_tuning = MultiProposalTuning(tunings))
            selected = Int[]
            for _ in expected_sequence
                state = BAT.mcmc_step!!(state)
                push!(selected, state.chain_state.proposal.active_idx)
            end
            @test selected == expected_sequence
            @test state.chain_state.nattempts == expected_attempts
        end

        user_schedule = [1, 1]
        owned_schedule = make_state(MCMCMultiProposal(random_walks, user_schedule);
            proposal_tuning = fixed_tuning)
        user_schedule .= [2, 0]
        @test owned_schedule.chain_state.proposal.picking_rule == [1, 1]
        push!(user_schedule, 1)
        @test length(owned_schedule.chain_state.proposal.picking_rule) == 2

        nested = multiprop([multiprop([HamiltonianMC()], [1]), RandomWalk()], [1, 1])
        invalid_configurations = [
            (MCMCMultiProposal(BAT.MCMCProposal[], Int[]), fixed_tuning, "at least one"),
            (MCMCMultiProposal(random_walks, [1]), fixed_tuning, "picking rule"),
            (MCMCMultiProposal(random_walks, [1, -1]), fixed_tuning, "nonnegative"),
            (MCMCMultiProposal(random_walks, [0, 0]), fixed_tuning, "positive mass"),
            (MCMCMultiProposal(random_walks, [1, typemax(Int)]), fixed_tuning,
                "cumulative"),
            (MCMCMultiProposal(random_walks, Categorical([1.0])), fixed_tuning, "picking rule"),
            (MCMCMultiProposal(BAT.MCMCProposal[HamiltonianMC(), RandomWalk()], [1, 1]),
                AdaptiveMultiPropTuning(), "categorical"),
            (nested, NoMCMCProposalTuning(), "nested"),
            (multiprop([RandomWalk(), HamiltonianMC()], [1, 1]),
                NoMCMCProposalTuning(), "component tuning"),
            (multiprop([RandomWalk()], [1]),
                BAT.StepSizeAdaptor(), "component tuning"),
            (MCMCMultiProposal(random_walks, [1, 1]), multituning(
                NoMCMCProposalTuning()), "component tunings"),
            (multiprop([HamiltonianMC()], [1]),
                multituning(NoMCMCProposalTuning()),
                "component tuning"),
            (multiprop([HamiltonianMC()], [1]),
                multituning(AdaptiveMultiPropTuning()),
                "component tuning"),
            (multiprop([RandomWalk()], [1]), multituning(BAT.StepSizeAdaptor()),
                "component tuning"),
        ]
        for (seed, (proposal, tuning, message)) in enumerate(invalid_configurations)
            err, rng_unchanged = construction_error(
                proposal; proposal_tuning = tuning, seed = 10 + seed,
            )
            @test err isa ArgumentError
            if err isa Exception
                @test occursin(message, lowercase(sprint(showerror, err)))
            end
            @test rng_unchanged
        end
    end

    @testset "scalar component dispatch and diagnostics" begin
        proposals = BAT.MCMCProposal[
            RandomWalk(proposaldist = Normal()),
            MCMCGlobalProposal(global_proposal = MvNormal(ones(1))),
            MALAProposal(),
            HamiltonianMC(step_size = 0.2, max_depth = 2),
        ]
        tunings = BAT.MCMCProposalTuning[
            NoMCMCProposalTuning(),
            NoMCMCProposalTuning(),
            BAT.StepSizeAdaptor(),
            BAT.StepSizeAdaptor(),
        ]
        state = make_state(
            MCMCMultiProposal(proposals, ones(Int, 4));
            proposal_tuning = multituning(tunings...),
            context = BATContext(rng = Philox4x((87234, 30)), ad = ForwardDiff),
        )
        BAT.mcmc_tuning_init!!(state, 5)
        for _ in 1:5
            state = BAT.mcmc_step!!(state)
        end

        @test state.chain_state.nattempts == [2, 1, 1, 1]
        @test getproperty.(state.proposal_tuner_state.proposal_tuners[3:4], :m) == [1, 1]

        components = BAT._mcmc_diagnostics_summary([state]).chain_diagnostics[1].components
        @test length(components) == 4
        @test getproperty.(components, :index) == 1:4
        @test getproperty.(components, :proposal_type) == [
            :RandomWalkProposalState,
            :MCMCGlobalProposalProposalState,
            :MALAProposalState,
            :HMCProposalState,
        ]
        @test getproperty.(components, :cycle_n_attempts) == [2, 1, 1, 1]
        @test getproperty.(components, :cycle_n_accepted) == state.chain_state.nsamples
        @test getproperty.(components, :cycle_acceptance_rate) ==
            state.chain_state.nsamples ./ state.chain_state.nattempts
        @test components[4].diagnostics.n_transitions == 1
    end

    @testset "shaped target" begin
        shaped_target = batmeasure(NamedTupleDist(a = Normal(), b = Normal()))
        samples = bat_sample(
            shaped_target,
            TransformedMCMC(
                proposal = multiprop([RandomWalk(), RandomWalk()], [1, 1]),
                pretransform = DoNotTransform(),
                adaptive_transform = BAT.NoAdaptiveTransform(),
                nchains = 1,
                nsteps = 4,
                convergence = AssumeConvergence(),
            ),
            BATContext(rng = Philox4x((87234, 31))),
        ).result
        @test first(samples).v isa NamedTuple{(:a, :b)}
    end

    @testset "multi-HMC warmup diagnostics" begin
        hmc = HamiltonianMC(step_size = 0.2, max_depth = 2)
        state = make_state(multiprop([hmc, hmc], [1, 1]);
            proposal_tuning = multituning(
                BAT.StepSizeAdaptor(), BAT.StepSizeAdaptor(),
            ))
        for _ in 1:4
            state = BAT.mcmc_step!!(state)
        end
        BAT.mcmc_mark_warmup_end!(state.chain_state.proposal)
        BAT.next_cycle!(state)
        state = BAT.mcmc_step!!(state)
        components = BAT._mcmc_diagnostics_summary([state]).chain_diagnostics[1].components
        @test getproperty.(components, :cycle_n_attempts) == [1, 0]
        @test getproperty.(components, :cycle_n_accepted) == state.chain_state.nsamples
        @test getproperty.(components, :cycle_acceptance_rate)[1] ==
            state.chain_state.nsamples[1]
        @test isnan(components[2].cycle_acceptance_rate)
        hmc_diagnostics = getproperty.(components, :diagnostics)
        @test getproperty.(getproperty.(hmc_diagnostics, :warmup), :n_transitions) == [2, 2]
        @test getproperty.(getproperty.(hmc_diagnostics, :sampling), :n_transitions) == [1, 0]
    end

    @testset "adaptive categorical ownership and component rates" begin
        random_walk = RandomWalk(
            target_acceptance = 0.9,
            target_acceptance_int = (0.8, 1.0),
        )
        user_rule = Categorical([0.5, 0.5])
        original_probabilities = copy(user_rule.p)
        state = make_state(multiprop([random_walk, random_walk], user_rule);
            proposal_tuning = AdaptiveMultiPropTuning())
        for (attempts, accepts, accept_prob) in [
            ([10, 10], [9, 1], [0.1, 0.9]),
            ([10, 0], [9, 0], [0.1, 0.9]),
        ]
            state.chain_state.nattempts .= attempts
            state.chain_state.nsamples .= accepts
            state.proposal_tuner_state.accept_prob .= accept_prob
            tuned = post_cycle(state)
            @test tuned.picking_rule.p == [0.6, 0.4]
        end
        @test state.chain_state.proposal.picking_rule.p == original_probabilities
        @test user_rule.p == original_probabilities
        post_step = BAT.mcmc_tune_proposal_post_step!!(
            state.chain_state.proposal,
            state.proposal_tuner_state,
            state.chain_state,
            BAT.MCMCStepInfo([1.0]),
        )[1]
        @test post_step.picking_rule.p != original_probabilities
        @test state.chain_state.proposal.picking_rule.p == original_probabilities
        @test user_rule.p == original_probabilities

        hmc_state = make_state(multiprop(
            [HamiltonianMC(), HamiltonianMC()], Categorical([0.5, 0.5]),
        ); proposal_tuning = AdaptiveMultiPropTuning())
        for (attempts, movements, accept_prob, expected_ranked, expected_finalized) in [
            ([10, 10], [1, 9], [0.9, 0.2], [0.6, 0.4], [1.0, 0.0]),
            ([10, 10], [9, 1], [0.2, 0.9], [0.4, 0.6], [0.0, 1.0]),
            ([10, 0], [1, 0], [0.9, 0.9], [0.6, 0.4], [1.0, 0.0]),
        ]
            hmc_state.chain_state.nattempts .= attempts
            hmc_state.chain_state.nsamples .= movements
            hmc_state.proposal_tuner_state.accept_prob .= accept_prob
            tuned_hmc = post_cycle(hmc_state)
            @test tuned_hmc.picking_rule.p == expected_ranked
            @test BAT.get_tuning_success(
                hmc_state.chain_state,
                hmc_state.chain_state.proposal,
                hmc_state.proposal_tuner_state,
            )
            finalized_hmc = BAT.mcmc_proposal_tuning_finalize!!(
                hmc_state.chain_state.proposal,
                hmc_state.proposal_tuner_state,
                hmc_state.chain_state,
            )[1]
            @test finalized_hmc.picking_rule.p == expected_finalized
        end
        hmc_state.chain_state.nattempts .= [10, 10]
        hmc_state.chain_state.nsamples .= [9, 9]
        hmc_state.proposal_tuner_state.accept_prob .= [0.2, 0.1]
        @test !BAT.get_tuning_success(
            hmc_state.chain_state,
            hmc_state.chain_state.proposal,
            hmc_state.proposal_tuner_state,
        )

        custom_proposal = BAT.MultiProposalState(
            BAT.MCMCProposalState[_CustomQualityProposalState()],
            Categorical([1.0]),
            1,
        )
        custom_tuner = BAT.AdaptiveMultiPropTunerState(0.1, 0.5, 0.8, [0.5])
        hmc_state.chain_state.nattempts .= [1, 0]
        custom_tuned = BAT.mcmc_tune_proposal_post_cycle!!(
            custom_proposal,
            custom_tuner,
            hmc_state.chain_state,
            DensitySampleVector[],
        )[1]
        @test custom_tuned.picking_rule.p == [1.0]
        @test BAT.get_tuning_success(
            hmc_state.chain_state, custom_proposal, custom_tuner,
        )
        custom_finalized = BAT.mcmc_proposal_tuning_finalize!!(
            custom_proposal, custom_tuner, hmc_state.chain_state,
        )[1]
        @test custom_finalized.picking_rule.p == [1.0]

        global_state = make_state(multiprop(
            [
                MCMCGlobalProposal(global_proposal = MvNormal(ones(1))),
                RandomWalk(),
            ],
            Categorical([0.5, 0.5]),
        ); proposal_tuning = AdaptiveMultiPropTuning())
        global_state.chain_state.nattempts .= [10, 10]
        global_state.chain_state.nsamples .= [0, 2]
        global_tuned = post_cycle(global_state)
        @test global_tuned.picking_rule.p == [0.4, 0.6]
        global_finalized = BAT.mcmc_proposal_tuning_finalize!!(
            global_state.chain_state.proposal,
            global_state.proposal_tuner_state,
            global_state.chain_state,
        )[1]
        @test global_finalized.picking_rule.p == [0.0, 1.0]

        zero_quality_rule = Categorical([0.8, 0.2])
        zero_quality_state = make_state(multiprop(
            [random_walk, random_walk], zero_quality_rule,
        ); proposal_tuning = AdaptiveMultiPropTuning())
        zero_quality_state.chain_state.nattempts .= 1
        zero_quality_state.proposal_tuner_state.accept_prob .= [0.1, 0.2]
        zero_quality_tuned = @test_logs (:warn, r"No proposal") begin
            post_cycle(zero_quality_state)
        end
        @test zero_quality_tuned.picking_rule.p == [0.5, 0.5]
        @test zero_quality_rule.p == [0.8, 0.2]
    end

    @testset "component returns" begin
        mala = MCMCMultiProposal(
            proposals = BAT.MCMCProposal[MALAProposal(), MALAProposal()],
            picking_rule = [1, 1],
        )
        mala_state = make_state(mala)
        for (i, tuner) in enumerate(mala_state.proposal_tuner_state.proposal_tuners)
            tuner.m = 1
            tuner.log_stepsize_bar = log(0.1 * i)
        end
        direct_stepsizes = [BAT.mcmc_proposal_tuning_finalize!!(
            mala_state.chain_state.proposal.proposal_states[i],
            mala_state.proposal_tuner_state.proposal_tuners[i],
            mala_state.chain_state,
        )[1].τ for i in 1:2]
        finalized = BAT.mcmc_proposal_tuning_finalize!!(
            mala_state.chain_state.proposal,
            mala_state.proposal_tuner_state,
            mala_state.chain_state,
        )[1]
        @test getproperty.(finalized.proposal_states, :τ) == direct_stepsizes

        mala_state.chain_state.nattempts .= 50
        mala_state.chain_state.nsamples .= 0
        for (proposal, tuner) in zip(
            finalized.proposal_states, mala_state.proposal_tuner_state.proposal_tuners,
        )
            tuner.run_nobs = 50
            tuner.min_run_nobs = 50
            tuner.run_accept_sum = 50 * BAT.get_target_acceptance_ratio(proposal)
        end
        @test BAT.get_tuning_success(
            mala_state.chain_state, finalized, mala_state.proposal_tuner_state,
        )

        multi_proposal = BAT.MultiProposalState(
            BAT.MCMCProposalState[_CountingProposalState(0), _CountingProposalState(0)],
            [1, 1],
            1,
        )
        multi_tuner = BAT.MultiProposalTunerState(
            BAT.MCMCProposalTunerState[_CountingTunerState(0), _CountingTunerState(0)]
        )
        chain_state = mala_state.chain_state
        proposals_new, tuners_new, chain_new = BAT.mcmc_proposal_tuning_finalize!!(
            multi_proposal, multi_tuner, chain_state,
        )
        @test getproperty.(proposals_new.proposal_states, :updates) == [1, 1]
        @test getproperty.(tuners_new.proposal_tuners, :updates) == [1, 1]
        @test chain_new.stepno == chain_state.stepno + 2

        proposals_new, tuners_new, chain_new = BAT.mcmc_tune_proposal_post_cycle!!(
            multi_proposal, multi_tuner, chain_state, DensitySampleVector[],
        )
        @test getproperty.(proposals_new.proposal_states, :updates) == [2, 2]
        @test getproperty.(tuners_new.proposal_tuners, :updates) == [2, 2]
        @test chain_new.stepno == chain_state.stepno + 2

        proposals_new, tuners_new, chain_new = BAT.mcmc_tune_proposal_post_step!!(
            multi_proposal, multi_tuner, chain_state, BAT.MCMCStepInfo([1.0]),
        )
        @test getproperty.(proposals_new.proposal_states, :updates) == [3, 2]
        @test getproperty.(tuners_new.proposal_tuners, :updates) == [3, 2]
        @test chain_new.stepno == chain_state.stepno + 1
    end

    @testset "component acceptance" begin
        random_walk = RandomWalk(target_acceptance = 1.0, target_acceptance_int = (0.9, 1.0))
        multi = MCMCMultiProposal(
            proposals = BAT.MCMCProposal[random_walk, random_walk],
            picking_rule = [3, 1],
        )
        fixed_state = make_state(multi)
        for _ in 1:4
            fixed_state = BAT.mcmc_step!!(fixed_state)
        end
        @test fixed_state.chain_state.nattempts == [3, 1]
        fixed_state.chain_state.nsamples .= [3, 1]
        @test BAT.detailed_eff_acceptance_ratio(fixed_state.chain_state) == [1.0, 1.0]

        BAT.next_cycle!(fixed_state)
        fixed_state = BAT.mcmc_step!!(fixed_state)
        @test fixed_state.chain_state.nattempts == [1, 0]
        fixed_state.chain_state.nsamples .= [1, 0]
        component_rates = BAT.detailed_eff_acceptance_ratio(fixed_state.chain_state)
        @test component_rates[1] == 1.0
        @test isnan(component_rates[2])
        @test !BAT.get_tuning_success(
            fixed_state.chain_state,
            fixed_state.chain_state.proposal,
            fixed_state.proposal_tuner_state,
        )

        adaptive_multi = MCMCMultiProposal(
            proposals = multi.proposals,
            picking_rule = Categorical([0.75, 0.25]),
        )
        adaptive_state = make_state(adaptive_multi; proposal_tuning = AdaptiveMultiPropTuning())
        adaptive_state.chain_state.nattempts .= [1, 0]
        adaptive_state.chain_state.nsamples .= [1, 0]
        @test BAT.get_tuning_success(
            adaptive_state.chain_state,
            adaptive_state.chain_state.proposal,
            adaptive_state.proposal_tuner_state,
        )
        adaptive_proposal = BAT.mcmc_proposal_tuning_finalize!!(
            adaptive_state.chain_state.proposal,
            adaptive_state.proposal_tuner_state,
            adaptive_state.chain_state,
        )[1]
        @test adaptive_proposal.picking_rule.p == [1.0, 0.0]

        ordinary_mala = make_state(multiprop(
            [
                MALAProposal(),
                RandomWalk(target_acceptance = 0.1, target_acceptance_int = (0.05, 0.15)),
            ],
            [1, 1],
        ); proposal_tuning = multituning(BAT.StepSizeAdaptor(), NoMCMCProposalTuning()))
        BAT.mcmc_tuning_init!!(ordinary_mala, 20)
        ordinary_mala.chain_state.nattempts .= [10, 10]
        ordinary_mala.chain_state.nsamples .= [6, 1]
        @test BAT.get_tuning_success(
            ordinary_mala.chain_state,
            ordinary_mala.chain_state.proposal,
            ordinary_mala.proposal_tuner_state,
        )
    end
end
