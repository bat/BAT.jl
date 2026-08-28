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

    function make_state(proposal; proposal_tuning = nothing)
        tuning_kw = isnothing(proposal_tuning) ? (;) : (; proposal_tuning)
        algorithm = TransformedMCMC(;
            proposal,
            pretransform = DoNotTransform(),
            nwalkers = 1,
            tuning_kw...,
        )
        context = BATContext(rng = Philox4x((87234, 1)), ad = ForwardDiff)
        return BAT.MCMCState(algorithm, target, 1, [zeros(1)], context)
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
    end
end
