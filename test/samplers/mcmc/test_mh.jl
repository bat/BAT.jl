# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes, ArraysOfArrays

@testset "MetropolisHastings" begin
    rng = bat_rng()
    target = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

    shaped_density = @inferred(convert(AbstractDensity, target))
    @test shaped_density isa BAT.DistributionDensity
    density = unshaped(shaped_density)
    @test density isa BAT.DistributionDensity

    algorithm = MetropolisHastings()
    nchains = 4
 
    @testset "MCMC iteration" begin
        v_init = bat_initval(rng, density, InitFromTarget()).result
        @test @inferred(MCMCIterator(deepcopy(rng), algorithm, density, 1, unshaped(v_init, varshape(density)))) isa BAT.MHIterator
        chain = @inferred(MCMCIterator(deepcopy(rng), algorithm, density, 1, unshaped(v_init, varshape(density)))) 
        samples = DensitySampleVector(chain)
        BAT.mcmc_iterate!(samples, chain, max_nsteps = 10^5, nonzero_weights = false)
        @test chain.stepno == 10^5
        @test minimum(samples.weight) == 0
        @test isapprox(length(samples), 10^5, atol = 20)
        @test length(samples) == sum(samples.weight)
        @test isapprox(mean(samples), [1, -1, 2], atol = 0.2)
        @test isapprox(cov(samples), cov(unshaped(target)), atol = 0.3)

        samples = DensitySampleVector(chain)
        BAT.mcmc_iterate!(samples, chain, max_nsteps = 10^3, nonzero_weights = true)
        @test minimum(samples.weight) == 1
    end
 
    @testset "MCMC tuning and burn-in" begin
        init_alg = MCMCChainPoolInit()
        tuning_alg = AdaptiveMHTuning()
        burnin_alg = MCMCMultiCycleBurnin()
        convergence_test = BrooksGelmanConvergence()
        strict = true
        nonzero_weights = false
        callback = (x...) -> nothing
        max_nsteps = 10^5

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
            max_nsteps = div(max_nsteps, length(chains)),
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
            shaped_density,
            MCMCSampling(
                mcalg = algorithm,
                trafo = NoDensityTransform(),
                nsteps = 10^5,
                store_burnin = true
            )
        ).result

        @test first(samples).info.chaincycle == 1

        samples = BAT.bat_sample(
            shaped_density,
            MCMCSampling(
                mcalg = algorithm,
                trafo = NoDensityTransform(),
                nsteps = 10^5,
                store_burnin = false
            )
        ).result

        @test first(samples).info.chaincycle >= 2

        @test samples.v isa ShapedAsNTArray
        @test isapprox(mean(unshaped.(samples)), [1, -1, 2], atol = 0.3)
        @test isapprox(cov(unshaped.(samples)), cov(unshaped(target)), atol = 0.4)
    end

    @testset "MCMC sampling in transformed space" begin
        prior = BAT.example_posterior().prior
        likelihood = (logdensity = v -> 0,)
        inner_posterior = PosteriorDensity(likelihood, prior)
        # Test with nested posteriors:
        posterior = PosteriorDensity(likelihood, inner_posterior)
        smpls = bat_sample(posterior, MCMCSampling(mcalg = MetropolisHastings(), trafo = PriorToGaussian())).result

        isapprox(mean(unshaped.(smpls)), mean(nestedview(rand(unshaped(prior).dist, 10^5))), rtol = 0.1)
        isapprox(cov(unshaped.(smpls)), cov(unshaped(prior).dist), rtol = 0.1)
    end
end
