# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using HypothesisTests
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

    @testset "sampling" begin
        function test_sampler(; 
            num_dims::Integer = 2,
            mcalg::HamiltonianMC=HamiltonianMC(), 
            nsteps::Integer=10^4, 
            trafo::BAT.AbstractDensityTransformTarget=PriorToGaussian(),
            burnin::MCMCBurninAlgorithm=MCMCMultiCycleBurnin(),
            strict::Bool=true,
            test_name::String
            )
            @testset "$test_name" begin
            
                sampling_algorithm = MCMCSampling(mcalg=mcalg, nsteps=nsteps, trafo=trafo, burnin=burnin, strict=strict)
                
                μ = rand(num_dims)
                σ = rand(num_dims) |> x -> abs.(x)
        
                normal_dists = Vector{Normal{eltype(σ)}}(undef, num_dims)
                for i in eachindex(normal_dists)
                    normal_dists[i] = Normal(μ[i], σ[i])
                end
        
                mvnormal = product_distribution(normal_dists)
        
                samples = bat_sample(mvnormal, sampling_algorithm)
                X = samples.result.v.data
        
                @testset "KS Test" begin
                    for i in 1:num_dims
                        @test pvalue(ExactOneSampleKSTest(unique(X[i, :]), normal_dists[i]), tail=:both) > 0.045
                    end
                end
    
                @testset "chains" begin
                    chains = samples.generator.chains
                    num_chains = @inferred(length(chains))

                    @test num_chains == sampling_algorithm.nchains

                    for chain in chains
                        chain_info = chain.info

                        @test @inferred(BAT.mcmc_info(chain)) == chain_info
                        @test chain_info.converged == true
                        @test chain_info.tuned == true
                    end
                end
            end
        end
        @testset "NUTS" begin
            tuning = BAT.StanHMCTuning(target_acceptance=0.90)
            mcalg = HamiltonianMC(tuning=tuning)
            test_sampler(num_dims=4, mcalg=mcalg, test_name="NUTS")
        end
        
        @testset "ahmc integrators" begin
            mcalg_jittered_lf = HamiltonianMC(integrator=BAT.JitteredLeapfrogIntegrator())
            test_sampler(num_dims=4, mcalg=mcalg_jittered_lf, test_name="jittered leapfrog")

            mcalg_tempered_lf = HamiltonianMC(integrator=BAT.TemperedLeapfrogIntegrator())
            test_sampler(num_dims=4, mcalg=mcalg_tempered_lf, test_name="tempered leapfrog")
        end

        @testset "ahmc metrics" begin
            mcalg_unit_euc = HamiltonianMC(metric=BAT.UnitEuclideanMetric())
            test_sampler(num_dims=4, mcalg=mcalg_unit_euc, test_name="unit euclidean")

            mcalg_dense_euc = HamiltonianMC(metric=BAT.DenseEuclideanMetric())
            test_sampler(num_dims=4, mcalg=mcalg_dense_euc, test_name="dense euclidean")
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
