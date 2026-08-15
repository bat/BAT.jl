# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, IntervalSets, ValueShapes
import Random123

@testset "MCMCChainPoolInit" begin
    target = unshaped(batmeasure(Normal()))

    @testset "delegates MCMC state viability" begin
        context = BATContext(rng = Random123.Philox4x((0, 0)))
        initvals = BAT.bat_ensemble_initvals(target, InitFromTarget(), 1, context)
        mcmc_state = BAT.MCMCState(
            TransformedMCMC(nchains = 1),
            target,
            1,
            unshaped.(initvals),
            deepcopy(context)
        )

        @test BAT.isvalidstate(mcmc_state)
        @test !BAT.isviablestate(mcmc_state.chain_state)
        @test !BAT.isviablestate(mcmc_state)
    end

    @testset "keeps the exact candidate count" begin
        init_result = BAT.mcmc_init!(
            TransformedMCMC(nchains = 2),
            target,
            MCMCChainPoolInit(init_tries_per_chain = 1..1, nsteps_init = 100),
            (_...) -> nothing,
            BATContext(rng = Random123.Philox4x((0, 0)))
        )

        @test length(init_result.mcmc_states) == 2
        @test length(init_result.outputs) == 2
    end

    @testset "selects one chain" begin
        init_result = BAT.mcmc_init!(
            TransformedMCMC(nchains = 1),
            target,
            MCMCChainPoolInit(init_tries_per_chain = 1..1, nsteps_init = 100),
            (_...) -> nothing,
            BATContext(rng = Random123.Philox4x((0, 0)))
        )

        @test length(init_result.mcmc_states) == 1
        @test length(init_result.outputs) == 1
    end

    @testset "clusters excess candidates" begin
        init_result = BAT.mcmc_init!(
            TransformedMCMC(nchains = 2),
            target,
            MCMCChainPoolInit(init_tries_per_chain = 2..2, nsteps_init = 100),
            (_...) -> nothing,
            BATContext(rng = Random123.Philox4x((0, 0)))
        )

        @test length(init_result.mcmc_states) == 2
        @test length(init_result.outputs) == 2
    end

    @testset "rejects stuck chains" begin
        stuck_target = unshaped(batmeasure(Uniform(-1e-300, 1e-300)))
        err = try
            BAT.mcmc_init!(
                TransformedMCMC(nchains = 2),
                stuck_target,
                MCMCChainPoolInit(init_tries_per_chain = 1..1, nsteps_init = 100),
                (_...) -> nothing,
                BATContext(rng = Random123.Philox4x((0, 0)))
            )
            nothing
        catch err
            err
        end

        @test err isa ErrorException
        @test sprint(showerror, err) == "Failed to generate 2 viable MCMC chain states"
    end

    @testset "reports failed clustering" begin
        @test BAT._check_cluster_convergence(true)
        @test_throws ErrorException("k-means clustering of MCMC chain states did not converge") BAT._check_cluster_convergence(false)
    end
end
