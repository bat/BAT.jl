
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
3
`n` must be a tuple `(nsteps, nchains, npartitions)`. `posterior` must be a uniform
distribution for each dimension.
"""
function bat_sample(
    posterior::PosteriorDensity,
    n::Tuple{Integer,Integer, Integer},
    algorithm::PartitionedSampling;
    exploration_kwargs::NamedTuple = (init_strategy = "tmp", burnin_strategy = "tmp"),
    sampling_kwargs::NamedTuple = (init_strategy = "tmp", burnin_strategy = "tmp"),
)
    @info "Generate Exploration Samples"
    @info "Construct KD-Tree"
    @info "Sample Parallel"
    @info "Combine Samples"

    # return (result = (...), info = (integral, uncert, cpu_time, wc_time, worker_id, sample_ind, param_bounds), part_tree = tree)
end
