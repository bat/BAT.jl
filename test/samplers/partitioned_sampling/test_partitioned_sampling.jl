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
        #Checks the reading was correct and data types are consistent
        #Tests of algorithm results in next test set
        @test ps.exploration_sampler.nsteps == 10^4
        @test ps.exploration_sampler.mcalg isa MetropolisHastings
        @test ps.exploration_sampler.nchains == 4
        @test ps.exploration_sampler.trafo isa NoDensityTransform
        @test results.exp_samples isa StructVector
    end
    @testset "Partitioning Algorithm" begin
        #Checks reading and data types
        @test ps.npartitions == 4
        @test ps.partitioner isa BAT.KDTreePartitioning#Only partitioner alg implemented now is KDTree 
        @test ps.partitioner.extend_bounds == true
        @test ps.partitioner.partition_dims == :auto
        @test results.cost_values isa Vector{Float64} #prevents Nan
        @test results.part_tree isa BAT.SpacePartTree
        @test results.part_tree.left_child isa BAT.SpacePartTree
        @test results.part_tree.right_child isa BAT.SpacePartTree
        @test results.part_tree.cut_coordinate isa Union{Float64, Float32, Float16}
        @test results.part_tree.terminated_leaf isa Bool
        @test results.part_tree.cut_axis isa Union{Int, Int128, Int64, Int32, Int16, Int8}
        @test results.part_tree.cost_part isa Union{Float64, Float32, Float16}
        @test results.part_tree.cost isa Union{Float64, Float32, Float16}
        @test results.part_tree.bounds isa Matrix{AbstractFloat}

        #Checks results
        part_tree_bounds = BAT.get_tree_par_bounds(results.part_tree)

        function point_within_bounds(bounds, array)
            #Checks if the point  is within the bounds in bounds matrix. Return a boolean Matrix
            boolean_matrix = trues(size(bounds)[1],size(bounds[1])[1])
            for (i, b) in enumerate(bounds)
                for (j, bi) in enumerate(eachrow(b))
                    boolean_value =  bi[1] <= array[j] & array[j] <= bi[2]
                    boolean_matrix[i, j] = boolean_value
                end
            end
            return boolean_matrix
        end

        # Iterate over all the elements of the array and compare them with the boundary matrix, 
        # return the index where the point is between the boundaries
        pos = zeros(Int8, size(μ, 1))
        for i=1:size(μ,1)
            a = point_within_bounds(part_tree_bounds, μ[i,:]);
            pos[i] = findall([a[i, :] == trues(3) for i=1:size(a,1)])[1]#In which supsace is the mode?
        end

        @test sort(pos) == 1:size(μ,1)# Are all the points assigned to different boundaries and they cover all the subdivisions?

    end
    @testset "Sampling Subspaces" begin
        posteriors_array = BAT.convert_to_posterior(posterior, results.part_tree, extend_bounds = true)#Partition Posterior

        @test posteriors_array isa Vector
        @test all(map(x -> x isa BAT.PosteriorDensity, posteriors_array))#Each partition should be a posterior
        @test typeof(posteriors_array[1]) == typeof(posteriors_array[2])#The subtypes should be exactly the same

        subspace = [1, posteriors_array[1], 10^3, MCMCSampling(mcalg = MetropolisHastings(), nsteps = 5*10^5, trafo = NoDensityTransform(),), 
        AHMIntegration()]#parameters
        samples_subspace = BAT.sample_subspace(subspace...)#function to test
        md_array = Array(samples_subspace.samples.v.a)
        md_array = hcat(md_array...)#2D array 3 x N(No samples)
        sample_mean = mean(md_array, dims = 2);#mean coordinate in 3D space

        tree_bounds = BAT.get_tree_par_bounds(results.part_tree)[1]# this always returns (-1,-1,-1), correct?
        @test isapprox(sample_mean[1], -1., atol = 1e-2)# test mean
        @test isapprox(sample_mean[2], -1., atol = 1e-2)
        @test isapprox(sample_mean[3], -1., atol = 1e-2)
        @test all(isapprox.(std(md_array, dims = 2).^2, 0.1, atol = 1e-1))#test std
        #test that the sum of weights should be equal to the integral of posterior
        @test isapprox(sum(samples_subspace.samples.weight), samples_subspace.info.density_integral[1].val)

        #test types
        @test samples_subspace isa NamedTuple
        @test keys(samples_subspace) == (:samples, :info)
        @test samples_subspace.samples isa StructVector
        @test all([t in sort(collect(propertynames(samples_subspace.samples))) for t in [:aux, :info, :logd, :v, :weight]])
        @test samples_subspace.info isa Table
    end
    @testset "Combinig Samples" begin
    
        @test isapprox(size(results.result)[1], 4*10^4, rtol = 1e-1)#Total number of samples approx 4*10^4
        @test size(results.info)[1] == 4#4 partitions
    end
end
