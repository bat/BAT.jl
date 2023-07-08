# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes, ArraysOfArrays, DensityInterface
using IntervalSets
using AutoDiffOperators
import ForwardDiff, Zygote

import AdvancedHMC

@testset "HamiltonianMC" begin
    context = BATContext(ad = ADModule(:ForwardDiff))
    target = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

    shaped_density = @inferred(convert(AbstractMeasureOrDensity, target))
    @test shaped_density isa BAT.DistMeasure
    density = unshaped(shaped_density)
    @test density isa BAT.DistMeasure

    algorithm = HamiltonianMC()
    nchains = 4
 
    @testset "MCMC iteration" begin
        v_init = bat_initval(density, InitFromTarget(), context).result
        # Note: No @inferred, since MCMCIterator is not type stable (yet) with HamiltonianMC
        @test BAT.MCMCIterator(algorithm, density, 1, unshaped(v_init, varshape(density)), deepcopy(context)) isa BAT.MCMCIterator
        chain = BAT.MCMCIterator(algorithm, density, 1, unshaped(v_init, varshape(density)), deepcopy(context))
        tuner = BAT.StanHMCTuning()(chain)
        nsteps = 10^4
        BAT.tuning_init!(tuner, chain, 0)
        BAT.tuning_reinit!(tuner, chain, div(nsteps, 10))
        samples = DensitySampleVector(chain)
        BAT.mcmc_iterate!(samples, chain, tuner, max_nsteps = nsteps, nonzero_weights = false)
        @test chain.stepno == nsteps
        @test minimum(samples.weight) == 0
        @test isapprox(length(samples), nsteps, atol = 20)
        @test length(samples) == sum(samples.weight)
        @test BAT.test_dist_samples(unshaped(target), samples)

        samples = DensitySampleVector(chain)
        BAT.mcmc_iterate!(samples, chain, max_nsteps = 10^3, nonzero_weights = true)
        @test minimum(samples.weight) == 1
    end

    @testset "MCMC tuning and burn-in" begin
        max_nsteps = 10^5
        tuning_alg = BAT.StanHMCTuning()
        trafo = DoNotTransform()
        init_alg = bat_default(MCMCSampling, Val(:init), algorithm, trafo, nchains, max_nsteps)
        burnin_alg = bat_default(MCMCSampling, Val(:burnin), algorithm, trafo, nchains, max_nsteps)
        convergence_test = BrooksGelmanConvergence()
        strict = true
        nonzero_weights = false
        callback = (x...) -> nothing

        # Note: No @inferred, not type stable (yet) with HamiltonianMC
        init_result = BAT.mcmc_init!(
            algorithm,
            density,
            nchains,
            init_alg,
            tuning_alg,
            nonzero_weights,
            callback,
            context
        )

        (chains, tuners, outputs) = init_result
        #@test chains isa AbstractVector{<:BAT.AHMCIterator}
        #@test tuners isa AbstractVector{<:BAT.AHMCTuner}
        #@test outputs isa AbstractVector{<:DensitySampleVector}

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
                nsteps = 10^4,
                store_burnin = true
            ),
            context
        ).result

        # ToDo: First HMC sample currently had chaincycle set to 0, should be fixed.
        # @test first(samples).info.chaincycle == 1
        @test samples[2].info.chaincycle == 1

        smplres = BAT.sample_and_verify(
            shaped_density,
            MCMCSampling(
                mcalg = algorithm,
                trafo = DoNotTransform(),
                nsteps = 10^4,
                store_burnin = false
            ),
            target,
            context
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
        @test BAT.sample_and_verify(posterior, MCMCSampling(mcalg = HamiltonianMC(), trafo = PriorToGaussian()), prior.dist, context).verified
    end

    @testset "HMC autodiff" begin
        posterior = BAT.example_posterior()

        for adsel in [ADModule(:ForwardDiff), ADModule(:Zygote)]
            @testset "$adsel" begin
                context = BATContext(ad = adsel)

                hmc_sampling_alg = MCMCSampling(
                    mcalg = HamiltonianMC(),
                    nchains = 2,
                    nsteps = 100,
                    init = MCMCChainPoolInit(init_tries_per_chain = 2..2, nsteps_init = 5),
                    burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 100, max_ncycles = 1),
                    strict = false
                )
                
                @test bat_sample(posterior, hmc_sampling_alg, context).result isa DensitySampleVector
            end
        end
    end
end
