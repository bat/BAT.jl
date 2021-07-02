# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
using Distributions, LinearAlgebra, ValueShapes


@testset "BAT_partitioned_sampling" begin
    σ = [0.1 0 0; 0 0.1 0; 0 0 0.1]
    μ = [1 1 1; -1 1 0; -1 -1 -1; 1 -1 0]
    mixture_model = MixtureModel(MvNormal[MvNormal(μ[i,:], Matrix(Hermitian(σ)) ) for i in 1:4])
    # ToDo: Use more complex prior with non-uniform distribution
    prior = NamedTupleDist(a = [Normal(0,2), Normal(0,2), Normal(0, 2)])
    likelihood = let model = mixture_model
        params -> LogDVal(logpdf(model, params.a))
    end

    posterior = PosteriorDensity(likelihood, prior)
    transformed_posterior, trafo = bat_transform(PriorToUniform(), posterior)

    #Sampling and integration algorithms
    mcmc = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 10^3);
    ahmi = ahmi = AHMIntegration(whitening = BAT.NoWhitening(), max_startingIDs = 10^3)
    #sobol = BAT.SobolSampler(nsamples = 500)
    mcmc_exp = MCMCSampling(mcalg = MetropolisHastings(), nsteps = 1000, nchains=20, strict=false)

    ps = PartitionedSampling(sampler = mcmc, npartitions=4, exploration_sampler=mcmc_exp, integrator = ahmi, nmax_resampling=5);


    #Sampling with space partition
    results = bat_sample(posterior , ps)
    
    #Kolmogorov-Smirnov Test
    #mcmc_samples = @inferred(bat_sample(posterior, mcmc))#sample from original pdf with MCMC
    #mcmc_samples =bat_sample(posterior, mcmc) #sample from original pdf with MCMC
    iid_distribution = NamedTupleDist(a = mixture_model,)
    iid_samples = bat_sample(iid_distribution, IIDSampling(nsamples=10^6))
    ks_test = bat_compare(iid_samples.result, results.result)#Run Kolmogorov-Smirnov test
    @test all(ks_test.result.ks_p_values .> 0.7)#Check that all the p-values are bigger than 0.7
    @testset "Array of Posteriors" begin
        posteriors_array = BAT.convert_to_posterior(transformed_posterior, results.part_tree, extend_bounds = true)

        @test posteriors_array isa AbstractVector{<:BAT.AbstractPosteriorDensity}
        
        #Creates a matrix whose rows represents subspaces, cols represents modes and whose entries are 1s or 0s
        #1 if mean is in the corresponding subspace, 0 otherwise
        position_matrix = [Base.in(u, post.parbounds.vol) for u in trafo.(varshape(posterior).(eachrow(μ))), post in posteriors_array]
        #Check that each mode is in a different subspace
        @test sort([index[1] for index in findall(x -> x == true, position_matrix)]) == 1:4

    end
    @testset "Combinig Samples" begin
        #Sum of the weights should be equal to sum of integrals
        @test isapprox(sum(results.info.density_integral).val, sum(results.result.weight), rtol = 1e-2)
        @test size(results.info)[1] == 4#4 partitions
    end
end
