# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    PartitionedSampling

*BAT-internal, not part of stable public API.*

The algorithm that partitions parameter space by multiple subspaces and
samples/integrates them independently (See arXiv reference).

The default constructor is using `MetropolisHastings` sampler,
`AHMIntegration` integrator, and `KDTreePartitioning`:

    PartitionedSampling()

"""
@with_kw struct PartitionedSampling{S<:AbstractSamplingAlgorithm,
    I<:IntegrationAlgorithm, P<:SpacePartitioningAlgorithm} <: AbstractSamplingAlgorithm
    sampler::S = MetropolisHastings()
    exploration_sampler::S = sampler
    partitioner::P = KDTreePartitioning()
    integrator::I = AHMIntegration()
    exploration_kwargs::NamedTuple = NamedTuple()
    sampling_kwargs::NamedTuple = NamedTuple()
    n_exploration::Tuple{Integer,Integer} = (10^2, 40)
end

export PartitionedSampling


"""
    function bat_sample(
        posterior::PosteriorDensity,
        n::Tuple{Integer,Integer, Integer},
        algorithm::PartitionedSampling;
    )

*BAT-internal, not part of stable public API.*

Sample partitioned `posterior` using sampler, integrator, and space
partitioning algorithm specified in `algorithm`. `n` must be a tuple
`(nsteps, nchains, npartitions)`. `posterior` must be a uniform
distribution for each dimension.
"""
function bat_sample(
        posterior::PosteriorDensity,
        n::Tuple{Integer,Integer, Integer},
        algorithm::PartitionedSampling
    )

    n_samples, n_chains, n_subspaces = n

    @info "Generating Exploration Samples"
    exploration_samples = bat_sample(posterior, algorithm.n_exploration, algorithm.exploration_sampler; algorithm.exploration_kwargs...).result

    @info "Construct Partition Tree"
    partition_tree, cost_values = partition_space(exploration_samples, n_subspaces, algorithm.partitioner)

    # ToDo: .....

    @info "Sample Parallel"
    #ToDo: Convert partition_tree -> set of posterior with corresponding bounded priors
    iterator_subspaces = [[subspace_ind, posterior, (n_samples, n_chains), algorithm.sampler, algorithm.integrator, algorithm.sampling_kwargs] for subspace_ind in Base.OneTo(n_subspaces)]
    samples_subspaces = pmap(inp -> sample_subspace(inp...), iterator_subspaces)

    @info "Combine Samples"
    for subspace in samples_subspaces[2:end]
        append!(samples_subspaces[1].samples, subspace.samples)
        append!(samples_subspaces[1].info, subspace.info)
    end

    return (result=samples_subspaces[1].samples,
                info = samples_subspaces[1].info,
                exp_samples = exploration_samples,
                part_tree = partition_tree,
                cost_values = cost_values
            )
end

function sample_subspace(
    space_id::Integer,
    posterior::PosteriorDensity,
    n::Tuple{Integer,Integer},
    sampling_algorithm::A,
    integration_algorithm::I,
    sampling_kwargs::N
) where {N<:NamedTuple, A<:AbstractSamplingAlgorithm, I<:IntegrationAlgorithm}

    samples_subspace = bat_sample(posterior, n, sampling_algorithm; sampling_kwargs...).result
    integras_subspace = bat_integrate(samples_subspace, integration_algorithm).result

    # α = exp(log(integras_subspace) + log(prior_normalization))
    α = exp(log(integras_subspace) + log(1.0))

    # samples_subspace.weight .= α.val .* samples_subspace.weight ./ sum(samples_subspace.weight)
    samples_subspace.weight .= 2 .* samples_subspace.weight # How to change weights to Float?

    info_subspace = TypedTables.Table(
            density_integral = [integras_subspace],
            sampling_time = [1.],
            integration_time = [2.0]
        )

    return (samples = samples_subspace, info = info_subspace)
end
