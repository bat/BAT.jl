# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes, ArraysOfArrays, DensityInterface

@testset "MetropolisHastings" begin
    rng = bat_rng()
    target = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

    shaped_density = @inferred(convert(AbstractMeasureOrDensity, target))
    @test shaped_density isa BAT.DistMeasure
    density = unshaped(shaped_density)
    @test density isa BAT.DistMeasure

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
        @test BAT.test_dist_samples(unshaped(target), samples)
    end

    @testset "bat_sample" begin
        samples = bat_sample(
            shaped_density,
            MCMCSampling(
                mcalg = algorithm,
                trafo = DoNotTransform(),
                nsteps = 10^5,
                store_burnin = true
            )
        ).result

        @test first(samples).info.chaincycle == 1

        smplres = BAT.sample_and_verify(
            shaped_density,
            MCMCSampling(
                mcalg = algorithm,
                trafo = DoNotTransform(),
                nsteps = 10^5,
                store_burnin = false
            ),
            target
        )
        samples = smplres.result
        @test first(samples).info.chaincycle >= 2
        @test samples.v isa ShapedAsNTArray
        @test smplres.verified
    end

    @testset "MCMC sampling in transformed space" begin
        prior = BAT.example_posterior().prior
        likelihood = logfuncdensity(v -> 0)
        inner_posterior = PosteriorMeasure(likelihood, prior)
        # Test with nested posteriors:
        posterior = PosteriorMeasure(likelihood, inner_posterior)
        @test BAT.sample_and_verify(posterior, MCMCSampling(mcalg = MetropolisHastings(), trafo = PriorToGaussian()), prior.dist).verified
    end
end
