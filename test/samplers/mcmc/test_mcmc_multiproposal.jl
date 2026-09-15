# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Distributions, Random123
import ForwardDiff
using Test

@testset "mcmc_multiproposal" begin
    @test BAT.sample_and_verify(
        Normal(),
        TransformedMCMC(
            proposal = MCMCMultiProposal(), pretransform = DoNotTransform(), nsteps = 2,
        ),
        Normal(), BATContext(rng = Philox4x((564, 533))), max_retries = 0,
    ).verified

    @test BAT.sample_and_verify(
        Normal(),
        TransformedMCMC(
            proposal = MCMCMultiProposal((RandomWalk(), MALAProposal()), [1, 1]),
            pretransform = DoNotTransform(), nsteps = 2,
        ),
        Normal(), BATContext(rng = Philox4x((564, 534)), ad = ForwardDiff), max_retries = 0,
    ).verified
end
