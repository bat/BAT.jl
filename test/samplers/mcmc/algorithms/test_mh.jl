# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes

@testset "MetropolisHastings" begin
    rng = bat_rng()
    target = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

    density = @inferred(convert(AbstractDensity, target))
    @test density isa BAT.DistributionDensity

    algorithm = MetropolisHastings()
    nchains = 4
 
    @testset "MCMC iteration" begin
        v_init = bat_initval(rng, density, InitFromTarget()).result
        @test @inferred(MCMCIterator(deepcopy(rng), algorithm, density, 1, unshaped(v_init, varshape(density)))) isa BAT.MHIterator
        chain = @inferred(MCMCIterator(deepcopy(rng), algorithm, density, 1, unshaped(v_init, varshape(density)))) 
        samples = DensitySampleVector(chain)
        BAT.mcmc_iterate!(samples, chain, max_nsamples = 10^5, max_nsteps = 10^5, nonzero_weights = false)
        @test chain.stepno == 10^5
        @test minimum(samples.weight) == 0
        @test isapprox(length(samples), 10^5, atol = 20)
        @test length(samples) == sum(samples.weight)
        @test isapprox(mean(samples), [1, -1, 2], atol = 0.2)
        @test isapprox(cov(samples), cov(unshaped(target)), atol = 0.3)

        samples = DensitySampleVector(chain)
        BAT.mcmc_iterate!(samples, chain, max_nsamples = 10^3, max_nsteps = 10^3, nonzero_weights = true)
        @test minimum(samples.weight) == 1
    end
 
    @testset "MCMC tuning and burn-in" begin
        n = 10^4
        init_alg = MCMCChainPoolInit()
        tuning_alg = AdaptiveMHTuning()
        burnin_alg = MCMCMultiCycleBurnin()
        convergence_test = BrooksGelmanConvergence()
        strict = true
        nonzero_weights = false
        callback = (x...) -> nothing
        max_neval = 10 * n
        max_time = Inf

        init_result = @inferred(BAT.mcmc_init!(
            rng,
            algorithm,
            density,
            nchains,
            init_alg,
            tuning_alg,
            nonzero_weights,
            callback,
        ))

        (chains, tuners, outputs) = init_result
        @test chains isa AbstractVector{<:BAT.MHIterator}
        @test tuners isa AbstractVector{<:BAT.ProposalCovTuner}
        @test outputs isa AbstractVector{<:DensitySampleVector}

        BAT.mcmc_burnin!(
            outputs,
            tuners,
            chains,
            burnin_alg,
            convergence_test,
            strict,
            nonzero_weights,
            callback
        )

        BAT.mcmc_iterate!(
            outputs,
            chains;
            max_nsamples = div(n, length(chains)),
            max_nsteps = div(max_neval, length(chains)),
            max_time = max_time,
            nonzero_weights = nonzero_weights,
            callback = callback
        )

        samples = DensitySampleVector(first(chains))
        append!.(Ref(samples), outputs)
        
        @test length(samples) == sum(samples.weight)
        @test isapprox(mean(samples), [1, -1, 2], atol = 0.3)
        @test isapprox(cov(samples), cov(unshaped(target)), atol = 0.4)
    end

    @testset "MCMC tuning and burn-in" begin
        samples = BAT.bat_sample(
            density, 10^4,
            MCMCSampling(
                sampler = algorithm,
                store_burnin = true
            )
        ).result

        @test first(samples).info.chaincycle == 1

        samples = BAT.bat_sample(
            density, 10^4,
            MCMCSampling(
                sampler = algorithm,
                store_burnin = false
            )
        ).result

        @test first(samples).info.chaincycle >= 2

        @test samples.v isa ShapedAsNTArray
        @test isapprox(mean(unshaped.(samples)), [1, -1, 2], atol = 0.3)
        @test isapprox(cov(unshaped.(samples)), cov(unshaped(target)), atol = 0.4)
    end
end
