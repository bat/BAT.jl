# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using BAT: NoMCMCProposalTuning, NoMCMCTransformTuning, NoAdaptiveTransform,
    TriangularAffineTransform, DiagonalAffineTransform, LowRankAffineTransform,
    StepSizeAdaptor, MCMCChainPoolInit, MCMCMultiCycleBurnin

# The TransformedMCMC keyword constructor resolves defaults through
# bat_default methods whose context-argument signatures must match the
# constructor's calls exactly - a mismatch does not error but silently
# falls through to the generic defaults. These tests pin the resolved
# defaults for every proposal family:
@testset "mcmc_defaults" begin
    @testset "HamiltonianMC" begin
        alg = TransformedMCMC(proposal = HamiltonianMC())
        @test alg.pretransform isa NormalBased
        @test alg.proposal_tuning isa StepSizeAdaptor
        @test alg.adaptive_transform isa TriangularAffineTransform
        @test alg.transform_tuning isa FisherTransformTuning
        @test alg.nwalkers == 1
        @test alg.nsteps == 10^4
        @test alg.init isa MCMCChainPoolInit
        @test alg.init.nsteps_init == 25
        @test alg.burnin isa MCMCMultiCycleBurnin
        @test alg.burnin.nsteps_per_cycle == max(div(alg.nsteps, 10), 250)
        @test alg.burnin.max_ncycles == 4
    end

    @testset "RandomWalk" begin
        alg = TransformedMCMC(proposal = RandomWalk())
        @test alg.pretransform isa NormalBased
        @test alg.proposal_tuning isa NoMCMCProposalTuning
        @test alg.adaptive_transform isa TriangularAffineTransform
        @test alg.transform_tuning isa RAMTuning
        @test alg.nsteps == 10^5
        @test alg.init isa MCMCChainPoolInit
        @test alg.init.nsteps_init == max(div(alg.nsteps, 100), 250)
        @test alg.burnin isa MCMCMultiCycleBurnin
        @test alg.burnin.nsteps_per_cycle == max(div(alg.nsteps, 10), 2500)
    end

    @testset "MALAProposal" begin
        alg = TransformedMCMC(proposal = BAT.MALAProposal())
        @test alg.pretransform isa NormalBased
        @test alg.proposal_tuning isa NoMCMCProposalTuning
        @test alg.adaptive_transform isa TriangularAffineTransform
        @test alg.transform_tuning isa FisherTransformTuning
        @test alg.nsteps == 10^5
    end

    @testset "MCMCGlobalProposal" begin
        alg = TransformedMCMC(proposal = MCMCGlobalProposal())
        @test alg.proposal_tuning isa NoMCMCProposalTuning
        @test alg.transform_tuning isa RAMTuning
        @test alg.nsteps == 10^5
    end

    @testset "transform tuning follows proposal and adaptive transform" begin
        alg_none = TransformedMCMC(proposal = RandomWalk(), adaptive_transform = NoAdaptiveTransform())
        @test alg_none.transform_tuning isa NoMCMCTransformTuning
        # No adaptive transform means no transform tuning, for HMC as well:
        alg_hmc_none = TransformedMCMC(proposal = HamiltonianMC(), adaptive_transform = NoAdaptiveTransform())
        @test alg_hmc_none.transform_tuning isa NoMCMCTransformTuning

        # The user selects the transform structure, gradient-based
        # proposals tune any of them via Fisher-divergence tuning:
        for at in (TriangularAffineTransform(), DiagonalAffineTransform(), LowRankAffineTransform())
            @test TransformedMCMC(proposal = HamiltonianMC(), adaptive_transform = at).transform_tuning isa FisherTransformTuning
            @test TransformedMCMC(proposal = BAT.MALAProposal(), adaptive_transform = at).transform_tuning isa FisherTransformTuning
        end

        # Non-gradient proposals have no diagonal/low-rank tuning:
        @test_throws ArgumentError TransformedMCMC(proposal = RandomWalk(), adaptive_transform = DiagonalAffineTransform())
        @test_throws ArgumentError TransformedMCMC(proposal = RandomWalk(), adaptive_transform = LowRankAffineTransform())
    end
end
