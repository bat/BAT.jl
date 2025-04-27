# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, ValueShapes, ArraysOfArrays, DensityInterface

@testset "RandomWalk" begin
    context = BATContext()
    objective = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

    shaped_target = @inferred(batmeasure(objective))
    @test shaped_target isa BAT.BATDistMeasure
    target = unshaped(shaped_target)
    @test target isa BAT.BATDistMeasure

    proposal = RandomWalk()
    nchains = 4
    nwalkers = 1

    samplingalg = TransformedMCMC(nchains = nchains, nwalkers = nwalkers)
 
    @testset "MCMC iteration" begin
        v_inits = BAT.bat_ensemble_initvals(target, InitFromTarget(), nwalkers, context)
        # TODO: MD, Reactivate type inference tests
        # @test @inferred(BAT.MCMCChainState(samplingalg, target, 1, unshaped.(v_inits), deepcopy(context))) isa BAT.MHChainState
        # chain = @inferred(BAT.MCMCChainState(samplingalg, target, 1, unshaped.(v_inits), deepcopy(context))) 
        mcmc_state = BAT.MCMCState(samplingalg, target, 1, unshaped.(v_inits), deepcopy(context))
        chain_output = BAT._empty_chain_outputs(mcmc_state)
        mcmc_state = BAT.mcmc_iterate!!(chain_output, mcmc_state; max_nsteps = 10^5, nonzero_weights = false)
        
        samples = BAT._empty_DensitySampleVector(mcmc_state)
        for walker_output in chain_output
            append!(samples, walker_output)
        end

        @test mcmc_state.chain_state.stepno == 10^5
        @test minimum(samples.weight) == 0
        @test isapprox(length(samples), 10^5, atol = 20)
        @test length(samples) == sum(samples.weight)
        @test isapprox(mean(samples), [1, -1, 2], atol = 0.2)
        @test isapprox(cov(samples), cov(unshaped(objective)), atol = 0.3)

        chain_output = BAT._empty_chain_outputs(mcmc_state)
        mcmc_state = BAT.mcmc_iterate!!(chain_output, mcmc_state; max_nsteps = 10^3, nonzero_weights = true)

        samples = BAT._empty_DensitySampleVector(mcmc_state)
        for walker_output in chain_output
            append!(samples, walker_output)
        end

        @test minimum(samples.weight) == 1
    end
 
    @testset "MCMC tuning and burn-in" begin
        init_alg = MCMCChainPoolInit()
        tuning_alg = AdaptiveAffineTuning()
        burnin_alg = MCMCMultiCycleBurnin()
        convergence_test = BrooksGelmanConvergence()
        strict = true
        nonzero_weights = false
        callback = (x...) -> nothing
        max_nsteps = 10^5

        samplingalg = TransformedMCMC(
            proposal = proposal,
            transform_tuning = tuning_alg,
            burnin = burnin_alg,
            nchains = nchains,
            nwalkers = nwalkers,
            convergence = convergence_test,
            strict = true,
            nonzero_weights = nonzero_weights
        )

        init_result = @inferred(BAT.mcmc_init!(
            samplingalg,
            target,
            init_alg,
            callback,
            context
        ))

        (mcmc_states, chain_outputs) = init_result

        # TODO: MD, Reactivate, for some reason fail
        # @test mcmc_states isa AbstractVector{<:BAT.MHChainState}
        # @test tuners isa AbstractVector{<:BAT.AdaptiveAffineTuningState}
        @test chain_outputs isa AbstractVector{<:AbstractVector{<:DensitySampleVector}}

        BAT.mcmc_burnin!(
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

        @test length(samples) == sum(samples.weight)
        @test BAT.test_dist_samples(unshaped(objective), samples)
    end

    @testset "bat_sample" begin
        samples = bat_sample(
            shaped_target,
            TransformedMCMC(
                proposal = proposal,
                pretransform = DoNotTransform(),
                store_burnin = true
            ),
            context
        ).result

        @test first(samples).info.chaincycle == 1

        smplres = BAT.sample_and_verify(
            shaped_target,
            TransformedMCMC(
                proposal = proposal,
                pretransform = DoNotTransform()
            ),
            objective
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
        @test BAT.sample_and_verify(posterior, TransformedMCMC(proposal = RandomWalk(), pretransform = PriorToNormal()), prior.dist).verified
    end
end
