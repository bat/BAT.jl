# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Distributions, LinearAlgebra, Random123
import ForwardDiff
using Test

@testset "mcmc_multiproposal" begin
    @testset "proposal selection and updates" begin
        proposals = (MALAProposal(), HamiltonianMC(), MALAProposal())
        tuning = BAT.StepSizeAdaptor()

        # The index of the active proposal in each step. Step size tuning
        # must change the state of that proposal and of no other:
        function selected_proposals(picking_rule, nsteps)
            algorithm = TransformedMCMC(
                proposal = MCMCMultiProposal(proposals, picking_rule),
                proposal_tuning = MultiProposalTuning((tuning, tuning, tuning)),
                pretransform = DoNotTransform(),
                adaptive_transform = BAT.NoAdaptiveTransform(),
                transform_tuning = BAT.NoMCMCTransformTuning(),
                nwalkers = 2,
            )
            context = BATContext(rng = Philox4x((564, 535)), ad = ForwardDiff)
            state = BAT.MCMCState(algorithm, batmeasure(MvNormal(zeros(2), I)), 1, [[0.0, 0.0], [1.0, 1.0]], context)
            steps = map(1:nsteps) do _
                old_states = state.chain_state.proposal.proposal_states
                state = BAT.mcmc_step!!(state)
                multi_proposal = state.chain_state.proposal
                changed = map(!==, old_states, multi_proposal.proposal_states)
                active_idx = multi_proposal.active_idx
                (active_idx, changed == ntuple(==(active_idx), length(proposals)))
            end
            @test all(last, steps)
            return first.(steps)
        end

        # An integer rule cycles through the proposals:
        @test selected_proposals([2, 1, 1], 8) == [1, 1, 2, 3, 1, 1, 2, 3]
        # A random rule selects every proposal over time:
        @test Set(selected_proposals(Categorical([0.2, 0.3, 0.5]), 100)) == Set(1:3)
    end

    @test BAT.sample_and_verify(
        Normal(),
        TransformedMCMC(
            proposal = MCMCMultiProposal(), pretransform = DoNotTransform(), nsteps = 10^4,
        ),
        Normal(), BATContext(rng = Philox4x((564, 533))), max_retries = 0,
    ).verified

    @test BAT.sample_and_verify(
        Normal(),
        TransformedMCMC(
            proposal = MCMCMultiProposal((RandomWalk(), MALAProposal()), [1, 1]),
            pretransform = DoNotTransform(), nsteps = 10^4,
        ),
        Normal(), BATContext(rng = Philox4x((564, 534)), ad = ForwardDiff), max_retries = 0,
    ).verified
end
