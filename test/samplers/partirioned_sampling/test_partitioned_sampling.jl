#Script contatining tests for Partitioned Sampling

#Import Libraries
using BAT
using Test
using Distributions, StatsBase, StructArrays
using IntervalSets, ValueShapes, TypedTables, Random, LinearAlgebra


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

#Sampling algorithms
mcmc = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^4, trafo = NoDensityTransform(),)#Subspace sampling
mcmc_exp = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^4, nchains=4, trafo = NoDensityTransform())#Exploratory samples
ps = PartitionedSampling(sampler = mcmc, npartitions=4, exploration_sampler=mcmc_exp)#Space Partitioner

#Sampling with space partition
results = bat_sample(posterior , ps)
############### End Preliminars ##################

#Tests

@testset "BAT_partitioned_sampling" begin
    @testset "Exploratory Phase" begin
        @test ps.exploration_sampler.nsteps == 10^4
    end
end
