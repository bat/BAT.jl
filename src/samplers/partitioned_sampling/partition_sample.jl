# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    PartitionedSampling

The algorithm that partitions parameter space by multiple subspaces and
samples/integrates them independently (See arXiv reference).

The default constructor is using `MetropolisHastings` sampler,
`AHMIntegration` integrator, and `KDTreePartitioning`:

    PartitionedSampling()

"""
@with_kw struct PartitionedSampling{S<:AbstractSamplingAlgorithm,
    I<:IntegrationAlgorithm, P<:SpacePartitioningAlgorithm} <: AbstractSamplingAlgorithm
    exploration_algm::S = MetropolisHastings()
    partiton_algm::P = KDTreePartitioning()
    sampling_algm::S = MetropolisHastings()
    integration_algm::I = AHMIntegration()
end

export PartitionedSampling


"""
    function bat_sample(
        posterior::PosteriorDensity,
        n::Tuple{Integer,Integer, Integer},
        algorithm::PartitionedSampling;
        n_subspaces::Integer,
        sampling_kwargs::NamedTuple,
    )

Sample partitioned `posterior` using sampler, integrator, and space
partitioning algorithm specified in `algorithm` with corresponding kwargs
given by `exploration_kwargs`, and `sampling_kwargs`.
`n` must be a tuple `(nsteps, nchains, npartitions)`. `posterior` must be a uniform
distribution for each dimension.
"""
function bat_sample(
    posterior::PosteriorDensity,
    n::Tuple{Integer,Integer, Integer},
    algorithm::PartitionedSampling;
    exploration_kwargs::NamedTuple = NamedTuple(),
    sampling_kwargs::NamedTuple = NamedTuple(),
    n_exploration::Tuple{Integer,Integer} = (10^2, 40)
)
    n_samples, n_chains, n_subspaces = n

    @info "Generating Exploration Samples"
    exploration_samples = bat_sample(posterior, n_exploration, algorithm.exploration_algm; exploration_kwargs...).result

    @info "Construct Partition Tree"
    partition_tree, cost_values = partition_space(exploration_samples, n_subspaces, algorithm.partiton_algm)

    @info "Sample Parallel"
    #ToDo: Convert partition_tree -> set of posterior with corresponding bounded priors
    iterator_subspaces = [[subspace_ind, posterior, (n_samples, n_chains), algorithm.sampling_algm, algorithm.integration_algm, sampling_kwargs] for subspace_ind in Base.OneTo(n_subspaces)]
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

    # ToDo: How to change MCMC Weights?

    info_subspace = TypedTables.Table(time_mcmc = [1.,], time_ahmi = [2.0,])

    return (samples = samples_subspace, info = info_subspace)
end
