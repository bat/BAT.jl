# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, LinearAlgebra, Random123
import ForwardDiff

using BAT: NoAdaptiveTransform, NoMCMCProposalTuning, NoMCMCTransformTuning


# An MCMC state whose walkers are stored in the order of `positions` and
# carry the given walker ids:
function walker_state(proposal, positions, walker_ids; proposal_tuning = NoMCMCProposalTuning())
    algorithm = TransformedMCMC(
        proposal = proposal,
        proposal_tuning = proposal_tuning,
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
    return state
end

function walker_paths(proposal, positions, walker_ids)
    state = walker_state(proposal, positions, walker_ids)
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


@testset "walker draws" begin
    positions = [[0.0, 0.0], [4.0, 4.0], [1.0, 2.0]]
    μ = batmeasure(MvNormal(zeros(2), I))

    # The variate and the acceptance uniform that each walker draws in a step:
    function draws_by_walker(proposal, positions, walker_ids)
        chain = walker_state(proposal, positions, walker_ids).chain_state
        draws = BAT._walker_draws(chain.proposal, chain, BAT._mcmc_step_rng(chain))
        variates, uniforms = BAT._walker_variates(draws, μ), BAT._walker_uniforms(draws)
        Dict(walker_ids[i] => (copy(variates[i]), uniforms[i]) for i in eachindex(walker_ids))
    end

    # RandomWalk declares batched draws, HamiltonianMC doesn't and so gets
    # one random stream per walker:
    for proposal in (RandomWalk(), HamiltonianMC(step_size = 0.2))
        reference = draws_by_walker(proposal, positions, [1, 2, 3])
        @test draws_by_walker(proposal, positions[[3, 1, 2]], [3, 1, 2]) == reference
        @test allunique(values(reference))
    end
end


@testset "HMC step size probes" begin
    positions = [[0.0, 0.0], [4.0, 4.0]]

    # The proposal after each of several transform changes, which make HMC
    # search for a new step size:
    function committed_proposals(proposal, tuning, positions, walker_ids)
        state = walker_state(proposal, positions, walker_ids; proposal_tuning = tuning)
        chain = state.chain_state
        map(1:10) do stepno
            chain.stepno = stepno
            first(BAT.mcmc_proposal_transform_committed!!(chain.proposal, state.proposal_tuner_state, chain))
        end
    end

    hmc, tuning = HamiltonianMC(), BAT.StepSizeAdaptor()
    stepsizes(args...) = [p.step_size for p in committed_proposals(hmc, tuning, args...)]
    @test stepsizes(reverse(positions), [2, 1]) == stepsizes(positions, [1, 2])

    # The HMC components of a multi-proposal probe with random streams of
    # their own:
    multi = MCMCMultiProposal((hmc, hmc), [1, 1])
    multi_tuning = MultiProposalTuning((tuning, tuning))
    component_stepsizes = [
        map(p -> p.step_size, multi_state.proposal_states)
        for multi_state in committed_proposals(multi, multi_tuning, positions, [1, 2])
    ]
    @test any(s -> s[1] != s[2], component_stepsizes)
end
