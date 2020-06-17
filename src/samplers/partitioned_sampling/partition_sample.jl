
"""
    IntegrationAlgorithm

Abstract type for integration algorithms.
"""
abstract type IntegrationAlgorithm end
export IntegrationAlgorithm


"""
    PartitionedSampling

The algorithm that partitions parameter space by multiple subspaces and
samples/integrates them independently (See arXiv reference).

The default constructor is using `MetropolisHastings` sampler,
`AHMIntegration` integrator, and `KDBinaryTree`:

    PartitionedSampling()

"""
struct PartitionedSampling{S<:AbstractSamplingAlgorithm,
    I<:IntegrationAlgorithm, P<:SpacePartitioningAlgorithm} <: AbstractSamplingAlgorithm
    samplingalgm::S
    integrationalgm::I
    partitonalgm::P
end

export PartitionedSampling

function PartitionedSampling()
    PartitionedSampling(MetropolisHastings(), AHMIntegration(), KDBinaryTree())
end
# export PartitionedSampling


"""
    function bat_sample(
        posterior::PosteriorDensity,
        n::Tuple{Integer,Integer},
        algorithm::PartitionedSampling;
        n_subspaces::Integer,
        sampling_kwargs::NamedTuple,
        integration_kwargs::NamedTuple,
        space_part_kwargs::NamedTuple
    )

Sample partitioned `posterior` using sampler, integrator, and space
partitioning algorithm specified in `algorithm` with corresponding kwargs
given by `sampling_kwargs`, `integration_kwargs`, and `space_part_kwargs`.

`n` must be a tuple `(nsteps, nchains)`. `posterior` must be a uniform
distribution for each dimension.
"""
function bat_sample(
    posterior::PosteriorDensity,
    n::Tuple{Integer,Integer},
    algorithm::PartitionedSampling;
    n_subspaces::Integer = 10,
    sampling_kwargs::NamedTuple = (init_strategy = "tmp", burnin_strategy = "tmp"),
    integration_kwargs::NamedTuple = (settings = "tmp",),
    space_part_kwargs::NamedTuple = (settings = "tmp",)
)
    @info "Generate Exploration Samples"
    @info "Construct KD-Tree"
    @info "Sample Parallel"
    @info "Combine Samples"

    # return (result = (...), info = (integral, uncert, cpu_time, wc_time, worker_id, sample_ind, param_bounds), part_tree = tree)
end
