# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Distributions
using Test

@testset "mcmc_multiproposal" begin
    default_proposal = MCMCMultiProposal()
    @test default_proposal.proposals == [RandomWalk()]
    @test default_proposal.picking_rule == Categorical([1.0])

    tuple_proposal = MCMCMultiProposal((RandomWalk(), RandomWalk()), [1, 2])
    @test tuple_proposal.proposals == [RandomWalk(), RandomWalk()]
    @test tuple_proposal.picking_rule == [1, 2]

    vector_proposal = MCMCMultiProposal([RandomWalk(), RandomWalk()], Categorical([1 / 3, 2 / 3]))
    @test vector_proposal.proposals == [RandomWalk(), RandomWalk()]
    @test vector_proposal.picking_rule == Categorical([1 / 3, 2 / 3])

    @test_throws ArgumentError MCMCMultiProposal(BAT.MCMCProposal[], Int[])
    @test_throws ArgumentError MCMCMultiProposal(proposals = BAT.MCMCProposal[])
    @test_throws ArgumentError MCMCMultiProposal((RandomWalk(),), [0])
    @test_throws ArgumentError MCMCMultiProposal((RandomWalk(),), [-1])
    @test_throws DimensionMismatch MCMCMultiProposal((RandomWalk(), RandomWalk()), [1])
    @test_throws DimensionMismatch MCMCMultiProposal((RandomWalk(), RandomWalk()), Categorical([1.0]))
end
