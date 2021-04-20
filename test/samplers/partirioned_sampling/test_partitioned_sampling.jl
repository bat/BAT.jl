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
    @testset "ExploratoryPhase" begin
        #Checks the reading was correct and data types are consistent
        #Tests of algorithm results in next test set
        @test ps.exploration_sampler.nsteps == 10^4
        @test ps.exploration_sampler.mcalg isa MetropolisHastings
        @test ps.exploration_sampler.nchains == 4
        @test ps.exploration_sampler.nchains == 4
        @test ps.exploration_sampler.trafo isa NoDensityTransform
        @test results.exp_samples isa StructVector
    end
    @testset "PartitioningAlgorithm" begin
        #Checks reading and data types
        @test ps.npartitions == 4
        @test ps.partitioner isa BAT.KDTreePartitioning#Only partitioner alg implemented now is KDTree 
        @test ps.partitioner.extend_bounds == true
        @test ps.partitioner.partition_dims == :auto

        #Tests of results
        part_tree_bounds = BAT.get_tree_par_bounds(results.part_tree)

        function point_within_bounds(bounds, array)
            #Checks if the point  is within any of the bounds in bounds matrix. Return a boolean Matrix
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
            pos[i] = findall([a[i, :] == trues(3) for i=1:size(a,1)])[1]
        end

        @test sort(pos) == 1:size(μ,1)
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
    end
    @testset "PosteriorsArray" begin
        posteriors_array = BAT.convert_to_posterior(posterior, results.part_tree, extend_bounds = true)

        @test posteriors_array isa Vector
        @test 
    end
end
