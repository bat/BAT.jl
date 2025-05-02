# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, ValueShapes, ArraysOfArrays, DensityInterface
using IntervalSets
import ForwardDiff, Zygote

import AdvancedHMC

@testset "HamiltonianMC" begin
    context = BATContext(ad = ForwardDiff)
    objective = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

    shaped_target = @inferred(batmeasure(objective))
    @test shaped_target isa BAT.BATDistMeasure
    target = unshaped(shaped_target)
    @test target isa BAT.BATDistMeasure

    proposal = HamiltonianMC()
    transform_tuning = StanLikeTuning()
    nchains = 4
    nwalkers = 1
    samplingalg = TransformedMCMC(proposal = proposal, transform_tuning = transform_tuning, nchains = nchains, nwalkers = nwalkers)

    @testset "MCMC iteration" begin
        v_inits = bat_ensemble_initvals(target, InitFromTarget(), nwalkers, context)
        # Note: No @inferred, since MCMCChainState is not type stable (yet) with HamiltonianMC
        @test BAT.MCMCChainState(samplingalg, target, 1, unshaped.(v_inits), deepcopy(context)) isa BAT.HMCChainState
        mcmc_state = BAT.MCMCState(samplingalg, target, 1, unshaped.(v_inits), deepcopy(context))
        nsteps = 10^4
        BAT.mcmc_tuning_init!!(mcmc_state, 0)
        BAT.mcmc_tuning_reinit!!(mcmc_state, div(nsteps, 10))

        walker_outputs = BAT._empty_chain_outputs(mcmc_state)
        mcmc_state = BAT.mcmc_iterate!!(walker_outputs, mcmc_state; max_nsteps = nsteps, nonzero_weights = false)
        @test mcmc_state.chain_state.stepno == nsteps

        samples = BAT._empty_DensitySampleVector(mcmc_state)
        for walker_output in walker_outputs
            append!(samples, walker_output)
        end

        @test minimum(samples.weight) == 0
        # @test isapprox(length(samples), nsteps, atol = 20) Hard to test with the new checked_push function avoiding duplicate samples
        @test sum(samples.weight) == mcmc_state.chain_state.stepno
        @test BAT.test_dist_samples(unshaped(objective), samples)

        walker_outputs = BAT._empty_chain_outputs(mcmc_state)
        mcmc_state = BAT.mcmc_iterate!!(walker_outputs, mcmc_state; max_nsteps = 10^3, nonzero_weights = true)

        samples = BAT._empty_DensitySampleVector(mcmc_state)
        for walker_output in walker_outputs
            append!(samples, walker_output)
        end

        @test minimum(samples.weight) == 1
    end

    @testset "MCMC tuning and burn-in" begin
        max_nsteps = 10^5
        transform_tuning = BAT.StanLikeTuning()
        pretransform = DoNotTransform()
        init_alg = bat_default(TransformedMCMC, Val(:init), proposal, pretransform, transform_tuning, nchains, nwalkers, max_nsteps)
        burnin_alg = bat_default(TransformedMCMC, Val(:burnin), proposal, pretransform, transform_tuning, nchains, nwalkers, max_nsteps)
        convergence_test = BrooksGelmanConvergence()
        strict = true
        nonzero_weights = false
        callback = (x...) -> nothing
    
        samplingalg = TransformedMCMC(proposal = proposal,
            transform_tuning = transform_tuning, 
            pretransform = pretransform, 
            nwalkers = nwalkers,
            init = init_alg, 
            burnin = burnin_alg, 
            convergence = convergence_test, 
            strict = strict, 
            nonzero_weights = nonzero_weights
        )
    
        # Note: No @inferred, not type stable (yet) with HamiltonianMC
        init_result = BAT.mcmc_init!(
            samplingalg,
            target,
            init_alg,
            callback,
            context
        )
    
        (mcmc_states, chain_outputs) = init_result
        @test mcmc_states isa AbstractVector{<:BAT.MCMCState}
        @test chain_outputs isa AbstractVector{<:AbstractVector{<:DensitySampleVector}}
    
        mcmc_states = BAT.mcmc_burnin!(
            chain_outputs,
            mcmc_states,
            samplingalg,
            callback
        )
    
        BAT.next_cycle!.(mcmc_states)
    
        mcmc_states = BAT.mcmc_iterate!!(
            chain_outputs,
            mcmc_states;
            max_nsteps = div(max_nsteps, length(mcmc_states)),
            nonzero_weights = nonzero_weights
        )
    
        samples = BAT._merge_chain_outputs(first(mcmc_states), chain_outputs)

        @test BAT.test_dist_samples(unshaped(objective), samples)
    end
    
    @testset "bat_sample" begin
        samples = bat_sample(
            shaped_target,
            TransformedMCMC(
                proposal = proposal,
                transform_tuning = StanLikeTuning(),
                pretransform = DoNotTransform(),
                nwalkers = nwalkers,
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
            TransformedMCMC(
                proposal = proposal,
                transform_tuning = StanLikeTuning(),
                pretransform = DoNotTransform(),
                nwalkers = nwalkers,
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
        trafo_samplingalg = TransformedMCMC(proposal = HamiltonianMC(), 
                                            transform_tuning = StanLikeTuning(), 
                                            pretransform = PriorToNormal(),
                                            nwalkers = nwalkers
                                           )
        @test BAT.sample_and_verify(posterior, trafo_samplingalg, prior.dist, context).verified
    end

    @testset "HMC autodiff" begin
        posterior = BAT.example_posterior()

        for admodule in [ForwardDiff, Zygote]
            @testset "$admodule" begin
                context = BATContext(ad = admodule)

                hmc_samplingalg = TransformedMCMC(
                    proposal = HamiltonianMC(),
                    transform_tuning = StanLikeTuning(),
                    nchains = 2,
                    nwalkers = nwalkers,
                    nsteps = 100,
                    init = MCMCChainPoolInit(init_tries_per_chain = 2..2, nsteps_init = 5),
                    burnin = MCMCMultiCycleBurnin(nsteps_per_cycle = 100, max_ncycles = 1),
                    strict = false
                )
                
                @test bat_sample(posterior, hmc_samplingalg, context).result isa DensitySampleVector
            end
        end
    end
end
