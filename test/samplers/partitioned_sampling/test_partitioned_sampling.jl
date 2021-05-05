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
    @testset "Array of Posteriors" begin
        posteriors_array = BAT.convert_to_posterior(posterior, results.part_tree, extend_bounds = true)#Partition Posterior

        @test posteriors_array isa AbstractVector{<:BAT.AbstractPosteriorDensity}
        
        #Creates a matrix whose rows represents subspaces, shoe cols represents means and whose entries are 1s or 0s
        #1 if mean is in the corresponding subspace, 0 otherwise
        position_matrix = [Base.in(u, post.parbounds.vol) for u in eachrow(μ), post in posteriors_array]
        #Check that each mode is in a different subspace
        @test sort([index[1] for index in findall(x -> x == true, position_matrix)]) == 1:4

    end
    @testset "Combinig Samples" begin
        #Sum of the weights should be equal to sum of integrals
        @test isapprox(sum(results.info.density_integral).val, sum(results.result.weight), rtol = 1e-2)
        @test size(results.info)[1] == 4#4 partitions
    end
end