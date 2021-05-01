#Script contatining tests for Partitioned Sampling

#Import Libraries
using BAT
using Test
using Distributions, LinearAlgebra, ValueShapes


############# Preliminars: define model, prior, likelihood, posterior, algorithms and run partitioned sampling ###############

## Define the model, a multimode gaussian will be taken here
σ = [0.1 0 0; 0 0.1 0; 0 0 0.1]
μ = [1 1 1; -1 1 0; -1 -1 -1; 1 -1 0]
mixture_model = MixtureModel(MvNormal[MvNormal(μ[i,:], Matrix(Hermitian(σ)) ) for i in 1:4])
#Define a flat prior
prior = NamedTupleDist(a = [Uniform(-50,50), Uniform(-50,50), Uniform(-50,50)])
#Define the Likelihood
likelihood = let model = mixture_model
    params -> LogDVal(logpdf(model, params.a))
end
#Define the posterior
posterior = PosteriorDensity(likelihood, prior)

#Sampling and integration algorithms
mcmc = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^3, trafo = NoDensityTransform(),);
#vegas = BAT.VEGASIntegration(trafo = NoDensityTransform(), rtol = 0.001, atol = 1.0e-8)
ahmi = AHMIntegration()
#sobol = BAT.SobolSampler(nsamples = 500, trafo = NoDensityTransform())
mcmc_exp = MCMCSampling(mcalg = MetropolisHastings(), nchains =4, nsteps = 400, trafo = NoDensityTransform(),);

ps = PartitionedSampling(sampler = mcmc, npartitions=4, exploration_sampler=mcmc_exp, integrator = ahmi);
############### End Preliminars ##################

#Tests

@testset "BAT_partitioned_sampling" begin
    #Sampling with space partition
    results = bat_sample(posterior , ps)
    
    #Kolmogorov-Smirnov Test
    mcmc_samples = @inferred(bat_sample(posterior, mcmc))#sample from original pdf with MCMC, is that what you wanted?
    ks_test = bat_compare(mcmc_samples.result, results.result)
    @test all(ks_test.result.ks_p_values .> 0.9)

    @testset "Partitioning Algorithm" begin
        #Checks results
        part_tree_bounds = @inferred(BAT.get_tree_par_bounds(results.part_tree))
        ## Important: with Sobol the modes are not always separated in different subspaces, with MCMC they are

        belongs_to_subpace = zeros(4,4)#Matrix where rows are subspaces and cols points in 3D space (modes)
        pos = zeros(4)#in which subspace a mode is in
        for (i, subspace) in enumerate(eachrow(part_tree_bounds))
            for (j, point) in enumerate(eachrow(μ))
                belongs_to_subpace[i,j] = all(subspace[1][:,2] .> point) &  all(subspace[1][:,1] .< point)# checks point is within bounds
                if belongs_to_subpace[i,j] == true#in which subspace a mode is in
                    pos[i] = j
                end
            end
        end

        @test sort(pos) == 1:size(μ,1)# Are all the points assigned to different boundaries and they cover all the subdivisions?

    end
    @testset "Array of Posteriors" begin
        posteriors_array = BAT.convert_to_posterior(posterior, results.part_tree, extend_bounds = true)#Partition Posterior

        @test posteriors_array isa AbstractVector{<:BAT.AbstractPosteriorDensity}
        #can I retrieve the samples from a subspace and compare them to theoretical values?
        #And create @test Sampling Subspaces
    end
    @testset "Combinig Samples" begin
        #Sum of the weights should be equal to sum of integrals
        @test isapprox(sum(results.info.density_integral).val, sum(results.result.weight), rtol = 1e-2)
        @test size(results.info)[1] == 4#4 partitions
    end
end