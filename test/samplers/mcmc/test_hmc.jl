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
    objective = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

    shaped_target = @inferred(batmeasure(objective))
    @test shaped_target isa BAT.BATDistMeasure
    target = unshaped(shaped_target)
    @test target isa BAT.BATDistMeasure

    proposal = HamiltonianMC()
    tuning = StanHMCTuning()
    nchains = 4
    sampling = MCMCSampling(proposal = proposal, tuning = tuning)

    @testset "MCMC iteration" begin
        v_init = bat_initval(target, InitFromTarget(), context).result
        # Note: No @inferred, since MCMCState is not type stable (yet) with HamiltonianMC
        # TODO: MD, reactivate
        @test BAT.MCMCState(sampling, target, 1, unshaped(v_init, varshape(target)), deepcopy(context)) isa BAT.HMCState
        mc_state = BAT.MCMCState(sampling, target, 1, unshaped(v_init, varshape(target)), deepcopy(context))
        tuner = BAT.StanHMCTuning()(mc_state)
        nsteps = 10^4
        BAT.tuning_init!(tuner, mc_state, 0)
        BAT.tuning_reinit!(tuner, mc_state, div(nsteps, 10))
        samples = DensitySampleVector(mc_state)
        mc_state, tuner, REMOVE_dummy_temperer = BAT.mcmc_iterate!!(samples, mc_state; tuner = tuner, max_nsteps = nsteps, nonzero_weights = false)
        @test mc_state.stepno == nsteps
        @test minimum(samples.weight) == 0
        @test isapprox(length(samples), nsteps, atol = 20)
        @test length(samples) == sum(samples.weight)
        @test BAT.test_dist_samples(unshaped(objective), samples)

        samples = DensitySampleVector(mc_state)
        mc_state, REMOVE_dummy_tuner, REMOVE_dummy_temperer = BAT.mcmc_iterate!!(samples, mc_state, max_nsteps = 10^3, nonzero_weights = true)
        @test minimum(samples.weight) == 1
    end

    @testset "MCMC tuning and burn-in" begin
        max_nsteps = 10^5
        tuning_alg = BAT.StanHMCTuning()
        trafo = DoNotTransform()
        init_alg = bat_default(MCMCSampling, Val(:init), proposal, trafo, nchains, max_nsteps)
        burnin_alg = bat_default(MCMCSampling, Val(:burnin), proposal, trafo, nchains, max_nsteps)
        convergence_test = BrooksGelmanConvergence()
        strict = true
        nonzero_weights = false
        callback = (x...) -> nothing

        sampling = MCMCSampling(proposal = proposal,
                                tuning = tuning_alg, 
                                pre_transform = trafo, 
                                init = init_alg, 
                                burnin = burnin_alg, 
                                convergence = convergence_test, 
                                strict = strict, 
                                nonzero_weights = nonzero_weights
        )

        # Note: No @inferred, not type stable (yet) with HamiltonianMC
        init_result = BAT.mcmc_init!(
            sampling,
            target,
            init_alg,
            callback,
            context
        )

        (mc_states, tuners, outputs) = init_result
        # @test mc_states isa AbstractVector{<:BAT.HMCState} # TODO: MD, reactivate, works for AbstractVector{<:MCMCState}, but doesn't seen to like the typealias 
        # @test tuners isa AbstractVector{<:BAT.HMCState}
        @test outputs isa AbstractVector{<:DensitySampleVector}

        BAT.mcmc_burnin!(
            outputs,
            tuners,
            mc_states,
            sampling,
            callback
        )

        mc_states, REMOVE_dummy_tuners, REMOVE_dummy_temperers = BAT.mcmc_iterate!!(
            outputs,
            mc_states;
            max_nsteps = div(max_nsteps, length(mc_states)),
            nonzero_weights = nonzero_weights
        )

        samples = DensitySampleVector(first(mc_states))
        append!.(Ref(samples), outputs)
        
        @test length(samples) == sum(samples.weight)
        @test BAT.test_dist_samples(unshaped(objective), samples)
    end

    @testset "bat_sample" begin
        samples = bat_sample(
            shaped_target,
            MCMCSampling(
                proposal = proposal,
                tuning = StanHMCTuning(),
                pre_transform = DoNotTransform(),
                nsteps = 10^4,
                store_burnin = true
            ),
            context
        ).result

        # ToDo: First HMC sample currently had chaincycle set to 0, should be fixed.
        # @test first(samples).info.chaincycle == 1
        @test samples[2].info.chaincycle == 1

        smplres = BAT.sample_and_verify(
            shaped_target,
            MCMCSampling(
                proposal = proposal,
                tuning = StanHMCTuning(),
                pre_transform = DoNotTransform(),
                nsteps = 10^4,
                store_burnin = false
            ),
            objective,
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
        @test BAT.sample_and_verify(posterior, MCMCSampling(proposal = HamiltonianMC(), tuning = StanHMCTuning(), pre_transform = PriorToGaussian()), prior.dist, context).verified
    end

    @testset "HMC autodiff" begin
        posterior = BAT.example_posterior()

        for adsel in [ADModule(:ForwardDiff), ADModule(:Zygote)]
            @testset "$adsel" begin
                context = BATContext(ad = adsel)

                hmc_sampling_alg = MCMCSampling(
                    proposal = HamiltonianMC(),
                    tuning = StanHMCTuning(),
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
