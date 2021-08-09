# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes, ArraysOfArrays

@testset "HamiltonianMC" begin
    rng = bat_rng()
    target = NamedTupleDist(a = Normal(1, 1.5), b = MvNormal([-1.0, 2.0], [2.0 1.5; 1.5 3.0]))

    shaped_density = @inferred(convert(AbstractDensity, target))
    @test shaped_density isa BAT.DistributionDensity
    density = unshaped(shaped_density)
    @test density isa BAT.DistributionDensity

    algorithm = HamiltonianMC()
    nchains = 4
 
    @testset "MCMC iteration" begin
        v_init = bat_initval(rng, density, InitFromTarget()).result
        # Note: No @inferred, since MCMCIterator is not type stable (yet) with HamiltonianMC
        @test MCMCIterator(deepcopy(rng), algorithm, density, 1, unshaped(v_init, varshape(density))) isa BAT.AHMCIterator
        chain = MCMCIterator(deepcopy(rng), algorithm, density, 1, unshaped(v_init, varshape(density)))
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
        @test BAT.likelihood_pvalue(unshaped(target), samples) > 10^-3

        samples = DensitySampleVector(chain)
        BAT.mcmc_iterate!(samples, chain, max_nsteps = 10^3, nonzero_weights = true)
        @test minimum(samples.weight) == 1
    end

    @testset "MCMC tuning and burn-in" begin
        max_nsteps = 10^5
        tuning_alg = BAT.StanHMCTuning()
        trafo = NoDensityTransform()
        init_alg = bat_default(MCMCSampling, Val(:init), algorithm, trafo, nchains, max_nsteps)
        burnin_alg = bat_default(MCMCSampling, Val(:burnin), algorithm, trafo, nchains, max_nsteps)
        convergence_test = BrooksGelmanConvergence()
        strict = true
        nonzero_weights = false
        callback = (x...) -> nothing

        # Note: No @inferred, not type stable (yet) with HamiltonianMC
        init_result = BAT.mcmc_init!(
            rng,
            algorithm,
            density,
            nchains,
            init_alg,
            tuning_alg,
            nonzero_weights,
            callback,
        )

        (chains, tuners, outputs) = init_result
        @test chains isa AbstractVector{<:BAT.AHMCIterator}
        @test tuners isa AbstractVector{<:BAT.AHMCTuner}
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
        @test BAT.likelihood_pvalue(unshaped(target), samples) > 10^-3
    end

    @testset "bat_sample" begin
        samples = bat_sample(
            shaped_density,
            MCMCSampling(
                mcalg = algorithm,
                trafo = NoDensityTransform(),
                nsteps = 10^4,
                store_burnin = true
            )
        ).result

        # ToDo: First HMC sample currently had chaincycle set to 0, should be fixed.
        # @test first(samples).info.chaincycle == 1
        @test samples[2].info.chaincycle == 1

        samples = bat_sample(
            shaped_density,
            MCMCSampling(
                mcalg = algorithm,
                trafo = NoDensityTransform(),
                nsteps = 10^4,
                store_burnin = false
            )
        ).result

        @test first(samples).info.chaincycle >= 2

        @test samples.v isa ShapedAsNTArray
        @test BAT.likelihood_pvalue(unshaped(target), unshaped.(samples)) > 10^-3
    end

    @testset "MCMC sampling in transformed space" begin
        prior = BAT.example_posterior().prior
        likelihood = (logdensity = v -> 0,)
        inner_posterior = PosteriorDensity(likelihood, prior)
        # Test with nested posteriors:
        posterior = PosteriorDensity(likelihood, inner_posterior)
        smpls = bat_sample(posterior, MCMCSampling(mcalg = HamiltonianMC(), trafo = PriorToGaussian())).result
        @test BAT.likelihood_pvalue(unshaped(prior.dist), unshaped.(smpls)) > 10^-3
    end
end
