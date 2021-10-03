# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using HypothesisTests
using IntervalSets
using StatsBase, Distributions, StatsBase, ValueShapes, ArraysOfArrays, DensityInterface

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
        @test BAT.dist_samples_pvalue(unshaped(target), samples) > 10^-3

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
        @test BAT.dist_samples_pvalue(unshaped(target), samples) > 10^-3
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

        smplres = BAT.sample_and_verify(
            shaped_density,
            MCMCSampling(
                mcalg = algorithm,
                trafo = NoDensityTransform(),
                nsteps = 10^4,
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
        inner_posterior = PosteriorDensity(likelihood, prior)
        # Test with nested posteriors:
        posterior = PosteriorDensity(likelihood, inner_posterior)
        @test BAT.sample_and_verify(posterior, MCMCSampling(mcalg = HamiltonianMC(), trafo = PriorToGaussian()), prior.dist).verified
    end

    @testset "sampling" begin
        function test_sampler(;
            num_dims::Integer = 2,
            sampling_algo=MCMCSampling(mcalg=HamiltonianMC()),
            test_name::String
            )
            @testset "$test_name" begin
                μ = rand(num_dims)
                σ = rand(num_dims) |> x -> abs.(x)
 
                normal_dists = Vector{Normal{eltype(σ)}}(undef, num_dims)
                for i in eachindex(normal_dists)
                    normal_dists[i] = Normal(μ[i], σ[i])
                end
        
                mvnormal = product_distribution(normal_dists)
        
                samples = bat_sample(mvnormal, sampling_algo)

                @testset "ahmi integration" begin
                    integral_res = bat_integrate(samples.result).result
                    integral_val = integral_res.val
                    integral_err = integral_res.err
        
                    @test isapprox(1, integral_val, rtol=7*integral_err)
                end

                @testset "likelihood pvalue test" begin
                    # Try one resampling if fail
                    if BAT.likelihood_pvalue(mvnormal, samples.result) < 0.05
                        samples = bat_sample(mvnormal, sampling_algo)
                    end
                    @test BAT.likelihood_pvalue(mvnormal, samples.result) >= 0.05
                end
    
                @testset "chains" begin
                    chains = samples.generator.chains
                    num_chains = @inferred(length(chains))

                    @test num_chains == sampling_algo.nchains

                    for chain in chains
                        chain_info = chain.info

                        @test @inferred(BAT.mcmc_info(chain)) == chain_info
                        @test chain_info.converged == true
                        @test chain_info.tuned == true
                    end
                end
            end
        end
        @testset "ahmc kernels" begin
            tuning = BAT.StanHMCTuning(target_acceptance=0.999)
            mcalg = HamiltonianMC(tuning=tuning)

            burnin = MCMCMultiCycleBurnin(max_ncycles=1000)
            chain_init = MCMCChainPoolInit(init_tries_per_chain=8..500, nsteps_init=2000)

            sampling_algo = MCMCSampling(nchains=8, trafo=BAT.NoDensityTransform(), init=chain_init, burnin=burnin, mcalg=mcalg)

            test_sampler(num_dims=2, sampling_algo=sampling_algo, test_name="NUTS")
        end
        
        @testset "ahmc integrators" begin
            tuning = BAT.StanHMCTuning(target_acceptance=0.999)
            burnin = MCMCMultiCycleBurnin(max_ncycles=1000, nsteps_final=2000)
            chain_init = MCMCChainPoolInit(init_tries_per_chain=8..600, nsteps_init=2000)

            mcalg_lf = HamiltonianMC(integrator=BAT.LeapfrogIntegrator(1.0), tuning=tuning)
            sampling_algo = MCMCSampling(nchains=8, init=chain_init, burnin=burnin, mcalg=mcalg_lf)
            test_sampler(num_dims=2, sampling_algo=sampling_algo, test_name="leapfrog")

            mcalg_jittered_lf = HamiltonianMC(integrator=BAT.JitteredLeapfrogIntegrator(), tuning=tuning)
            sampling_algo = MCMCSampling(nchains=8, init=chain_init, burnin=burnin, mcalg=mcalg_jittered_lf)
            test_sampler(num_dims=2, sampling_algo=sampling_algo, test_name="jittered leapfrog")

            mcalg_tempered_lf = HamiltonianMC(integrator=BAT.TemperedLeapfrogIntegrator(), tuning=tuning)
            sampling_algo = MCMCSampling(nchains=8, init=chain_init, burnin=burnin, mcalg=mcalg_tempered_lf)
            test_sampler(num_dims=2, sampling_algo=sampling_algo, test_name="tempered leapfrog")
        end

        @testset "ahmc metrics" begin
            tuning = BAT.StanHMCTuning(target_acceptance=0.999)

            burnin = MCMCMultiCycleBurnin(max_ncycles=1000, nsteps_final=2000)
            chain_init = MCMCChainPoolInit(init_tries_per_chain=8..600, nsteps_init=2000)

            mcalg_unit_euc = HamiltonianMC(metric=BAT.UnitEuclideanMetric(), tuning=tuning)
            sampling_algo = MCMCSampling(nchains=8, init=chain_init, burnin=burnin, mcalg=mcalg_unit_euc)
            test_sampler(num_dims=2, sampling_algo=sampling_algo, test_name="unit euclidean")

            mcalg_dense_euc = HamiltonianMC(metric=BAT.DenseEuclideanMetric(), tuning=tuning)
            sampling_algo = MCMCSampling(init=chain_init, burnin=burnin, mcalg=mcalg_dense_euc)
            test_sampler(num_dims=2, sampling_algo=sampling_algo, test_name="dense euclidean")
        end

        @testset "ahmc adaptors" begin
            burnin = MCMCMultiCycleBurnin(max_ncycles=1000, nsteps_final=2000)
            chain_init_more = MCMCChainPoolInit(init_tries_per_chain=8..1200, nsteps_init=2000)
            chain_init_most = MCMCChainPoolInit(init_tries_per_chain=8..600, nsteps_init=2000)

            mcalg_massmat = HamiltonianMC(tuning=BAT.MassMatrixAdaptor(0.999))
            sampling_algo = MCMCSampling(nchains=8, init=chain_init_most, burnin=burnin, trafo=BAT.NoDensityTransform(), mcalg=mcalg_massmat)
            test_sampler(num_dims=2, sampling_algo=sampling_algo, test_name="mass matrix")

            mcalg_stepsize = HamiltonianMC(tuning=BAT.StepSizeAdaptor())
            sampling_algo = MCMCSampling(nchains=8, init=chain_init_more, trafo=BAT.NoDensityTransform(), mcalg=mcalg_stepsize)
            test_sampler(num_dims=2, sampling_algo=sampling_algo, test_name="step size")
        end
    end

    @testset "ahmc sampleid" begin
        mvnorm = @inferred(product_distribution([Normal(), Normal()]))
        sampling_algo = @inferred(MCMCSampling(mcalg=HamiltonianMC(), nchains=2, trafo=NoDensityTransform(), nsteps=10^3))
    
        # Use @inferred when type stable
        samples_1 = bat_sample(mvnorm, sampling_algo).result
        samples_2 = bat_sample(mvnorm, sampling_algo).result
    
        id_vector_1 = samples_1.info
        id_vector_2 = samples_2.info
        id_vector_12 = @inferred(merge(id_vector_1, id_vector_2))
    
        num_sample_ids_1 = @inferred(length(id_vector_1))
        num_sample_ids_2 = @inferred(length(id_vector_2))
    
        @test @inferred(length(id_vector_12)) == num_sample_ids_1 + num_sample_ids_2
    
        @test id_vector_12[1:num_sample_ids_1] == id_vector_1
        @test id_vector_12[num_sample_ids_1+1:end] == id_vector_2
    
        empty_sample_idvec = @inferred(BAT.MCMCSampleIDVector())
        @test @inferred(isempty(empty_sample_idvec))
    
        merge!(id_vector_1, id_vector_2)
        @test id_vector_1 == id_vector_12
    end
end
