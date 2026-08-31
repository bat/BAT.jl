# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, LinearAlgebra, Random123
import ForwardDiff

using BAT: NoAdaptiveTransform, NoMCMCProposalTuning, NoMCMCTransformTuning


function walker_paths(proposal, positions, walker_ids)
    algorithm = TransformedMCMC(
        proposal = proposal,
        proposal_tuning = NoMCMCProposalTuning(),
        pretransform = DoNotTransform(),
        adaptive_transform = NoAdaptiveTransform(),
        transform_tuning = NoMCMCTransformTuning(),
        tempering = BAT.NoMCMCTempering(),
        nwalkers = length(walker_ids),
        nonzero_weights = false,
    )
    state = BAT.MCMCState(
        algorithm,
        batmeasure(MvNormal(zeros(2), I)),
        1,
        positions,
        BATContext(rng = Philox4x((0x0564, 4)), ad = ForwardDiff),
    )
    chain = state.chain_state
    for i in eachindex(walker_ids)
        info = chain.current.x.info[i]
        chain.current.x.info[i] = BAT.MCMCSampleID(
            info.chainid,
            Int32(walker_ids[i]),
            info.chaincycle,
            info.stepno,
            info.proposalid,
            info.sampletype,
        )
    end
    chain.walker_order .= sortperm(walker_ids)

    outputs = BAT._empty_chain_outputs(state)
    BAT.mcmc_iterate!!(outputs, state; max_nsteps = 3, nonzero_weights = false)
    Dict(walker_ids[i] => copy.(outputs[i].v) for i in eachindex(walker_ids))
end


@testset "MCMC walker storage order" begin
    positions = [[0.0, 0.0], [4.0, 4.0]]
    for proposal in (
        RandomWalk(proposaldist = Normal()),
        MCMCGlobalProposal(global_proposal = MvNormal(zeros(2), I)),
        MALAProposal(τ_base = 0.2),
        HamiltonianMC(step_size = 0.2, max_depth = 2),
    )
        reference = walker_paths(proposal, positions, [1, 2])
        permuted = walker_paths(proposal, reverse(positions), [2, 1])
        @test permuted == reference
    end
end
