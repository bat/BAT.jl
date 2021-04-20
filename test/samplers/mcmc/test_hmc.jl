# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra
using StatsBase, Distributions, StatsBase, ValueShapes, ArraysOfArrays
using IntervalSets

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
        tuning_alg = MCMCNoOpTuning()
        burnin_alg = MCMCMultiCycleBurnin()
        convergence_test = BrooksGelmanConvergence()
        strict = true
        nonzero_weights = false
        callback = (x...) -> nothing
        max_nsteps = 10^5

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
        @test tuners isa AbstractVector{<:BAT.MCMCNoOpTuner}
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
        @test isapprox(mean(unshaped.(samples)), [1, -1, 2], atol = 0.3)
        @test isapprox(cov(unshaped.(samples)), cov(unshaped(target)), atol = 0.4)
    end

    @testset "ahmc_adaptor" begin
        function test_ahmc_adaptor(target, adaptor::BAT.HMCAdaptor; 
            nchains::Integer=4, init::MCMCInitAlgorithm=MCMCChainPoolInit(), burnin::MCMCBurninAlgorithm=MCMCMultiCycleBurnin(),
            strict::Bool=true, testset_name::String, atol::Real=0.05)
            @testset "$testset_name" begin
                target_mean = [mean(getproperty(target, p)) for p in propertynames(target)]

                mcalg = HamiltonianMC(adaptor=adaptor)
                algorithm = MCMCSampling(mcalg=mcalg, nchains=nchains, store_burnin=false, init=init, burnin=burnin, strict=strict)

                out = bat_sample(target, algorithm)
                chains = out.generator.chains
                density_sample_vector = out.result

                for chain in chains
                    @test chain.algorithm.adaptor == adaptor
                end

                sample_mean = @inferred(mean(density_sample_vector))[1]
                for i in eachindex(target_mean)
                    @test isapprox(sample_mean[i], target_mean[i], atol=atol)
                end
            end
        end
        #init = MCMCChainPoolInit(init_tries_per_chain=8..500)
        #burnin = MCMCMultiCycleBurnin(nsteps_per_cycle=20000, max_ncycles=100)
        #test_ahmc_adaptor(target, BAT.NoAdaptor(), init=init, burnin=burnin, testset_name="NoAdaptor")
        #test_ahmc_adaptor(target, BAT.MassMatrixAdaptor(), init=init, burnin=burnin, testset_name="MassMatrixAdaptor")
        test_ahmc_adaptor(target, BAT.StepSizeAdaptor(), testset_name="StepSizeAdaptor")
        test_ahmc_adaptor(target, BAT.NaiveHMCAdaptor(), testset_name="NaiveHMCAdaptor")
    end

    @testset "ahmc_integrator" begin
        function test_ahmc_integrator(target, integrator::BAT.HMCIntegrator; nchains::Integer=4, strict::Bool=true, testset_name::String, atol::Real=0.05)
            @testset "$testset_name" begin
                target_mean = [mean(getproperty(target, p)) for p in propertynames(target)]

                mcalg = @inferred(HamiltonianMC(integrator=integrator))

                @test mcalg.integrator == integrator

                algorithm = @inferred(MCMCSampling(mcalg=mcalg, nchains=nchains, store_burnin=false, strict=strict))

                out = bat_sample(target, algorithm)
                chains = out.generator.chains
                density_sample_vector = out.result

                for chain in chains
                    @test chain.algorithm.integrator == integrator
                end

                sample_mean = @inferred(mean(density_sample_vector))[1]
                for i in eachindex(target_mean)
                    @test isapprox(sample_mean[i], target_mean[i], atol=atol)
                end
            end
        end

        test_ahmc_integrator(target, BAT.JitteredLeapfrogIntegrator(), testset_name="JitteredLeapfrogIntegrator")
        test_ahmc_integrator(target, BAT.TemperedLeapfrogIntegrator(), testset_name="TemperedLeapfrogIntegrator")
    end

    @testset "ahmc_sampleid" begin
        mvnorm = @inferred(product_distribution([Normal(), Normal()]))
        sampling_method = @inferred(MCMCSampling(mcalg=HamiltonianMC(), nchains=2, nsteps=10^3))
    
        samples_1 = bat_sample(mvnorm, sampling_method).result
        samples_2 = bat_sample(mvnorm, sampling_method).result
    
        id_vector_1 = samples_1.info
        id_vector_2 = samples_2.info

        @test id_vector_1 isa BAT.AHMCSampleIDVector
        @test id_vector_2 isa BAT.AHMCSampleIDVector

        id_vector_12 = @inferred(merge(id_vector_1, id_vector_2))
    
        num_sample_ids_1 = @inferred(length(id_vector_1))
        num_sample_ids_2 = @inferred(length(id_vector_2))
    
        @test @inferred(length(id_vector_12)) == num_sample_ids_1 + num_sample_ids_2
    
        @test id_vector_12[1:num_sample_ids_1] == id_vector_1
        @test id_vector_12[num_sample_ids_1+1:end] == id_vector_2
    
        @test @inferred(isempty(@inferred(BAT.MCMCSampleIDVector())))

        merge!(id_vector_1, id_vector_2)
        @test id_vector_1 == id_vector_12
    end

    @testset "ahmc_metric" begin
        function test_metric_mass(sample_generator::BAT.MCMCSampleGenerator, metric::BAT.UnitEuclideanMetric)
            @testset "mass matrix" begin
                for chain in sample_generator.chains
                    mass_inv = chain.hamiltonian.metric.M⁻¹
                    @test isdiag(mass_inv)
                end
            end
        end

        function test_metric_mass(sample_generator::BAT.MCMCSampleGenerator, metric::BAT.DenseEuclideanMetric)
            @testset "mass matrix" begin
                for chain in sample_generator.chains
                    mass_inv = chain.hamiltonian.metric.M⁻¹
                    sparsity = sum(iszero.(mass_inv)) / length(mass_inv)
                    @test sparsity == 0
                end
            end
        end

        function test_euclidean_metric(dist::BAT.AnySampleable; metric::BAT.HMCMetric, nchains::Integer=2, testset_name::String, rtol::Real=0.1)
            @testset "$testset_name" begin
                mcalg = HamiltonianMC(metric=metric)
                algorithm = MCMCSampling(mcalg=mcalg, store_burnin=false, nchains=nchains)

                s = bat_sample(dist, algorithm)

                for chain in s.generator.chains
                    @test size(chain.hamiltonian.metric) == size(dist) # dense has no size property
                    @test chain.algorithm.metric == metric
                end

                test_metric_mass(s.generator, metric)

                @test @inferred(isapprox(mean(s.result.v), mean(dist), rtol=rtol))
                @test @inferred(isapprox(var(s.result.v), var(dist), rtol=rtol))
            end
        end

        dist = product_distribution([Beta(2,2), Normal(-4, 5.)])

        test_euclidean_metric(dist, metric=BAT.UnitEuclideanMetric(), testset_name="UnitEuclideanMetric")
        test_euclidean_metric(dist, metric=BAT.DenseEuclideanMetric(), testset_name="DenseEuclideanMetric")
    end
    
    @testset "MCMC sampling in transformed space" begin
        prior = BAT.example_posterior().prior
        likelihood = (logdensity = v -> 0,)
        inner_posterior = PosteriorDensity(likelihood, prior)
        # Test with nested posteriors:
        posterior = PosteriorDensity(likelihood, inner_posterior)
        smpls = bat_sample(posterior, MCMCSampling(mcalg = HamiltonianMC(), trafo = PriorToGaussian())).result

        @test isapprox(mean(unshaped.(smpls)), mean(nestedview(rand(unshaped(prior).dist, 10^5))), rtol = 0.1)
        @test isapprox(cov(unshaped.(smpls)), cov(unshaped(prior).dist), rtol = 0.1)
    end
end
