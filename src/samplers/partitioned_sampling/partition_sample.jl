# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct PartitionedSampling <: AbstractSamplingAlgorithm

*Experimental feature, not part of stable public API.*

A sampling algorithm that partitions parameter space into multiple subspaces and
samples/integrates them independently ([Caldwell et al.](https://arxiv.org/abs/2008.03098)).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct PartitionedSampling{
    TR<:AbstractDensityTransformTarget,
    S<:AbstractSamplingAlgorithm,
    E<:AbstractSamplingAlgorithm,
    I<:IntegrationAlgorithm,
    P<:SpacePartitioningAlgorithm
} <: AbstractSamplingAlgorithm
    trafo::TR = PriorToUniform()
    npartitions::Integer = 10
    sampler::S = MCMCSampling()
    exploration_sampler::E = MCMCSampling(nchains=30)
    partitioner::P = KDTreePartitioning()
    integrator::I = AHMIntegration()
end

export PartitionedSampling


function bat_sample_impl(rng::AbstractRNG, target::PosteriorDensity, algorithm::PartitionedSampling)
    posterior = target

    @info "Generating Exploration Samples"
    exploration_samples = bat_sample(posterior, algorithm.exploration_sampler).result

    @info "Constructing Partition Tree"
    partition_tree, cost_values = partition_space(exploration_samples, algorithm.npartitions, algorithm.partitioner)
    # Convert 'partition_tree' structure into a set of truncated posteriors:
    posteriors_array = convert_to_posterior(posterior, partition_tree, extend_bounds = algorithm.partitioner.extend_bounds)

    @info "Sampling Subspaces"
    iterator_subspaces = [
        [subspace_ind, posteriors_array[subspace_ind],
        n_samples,
        algorithm.sampler,
        algorithm.integrator,
        algorithm.sampling_kwargs] for subspace_ind in Base.OneTo(n_subspaces)]
    samples_subspaces = pmap(inp -> sample_subspace(inp...), iterator_subspaces)

    @info "Combining Samples"
    # Save indices from different subspaces:
    samples_subspaces[1].info.samples_ind[1] = 1:length(samples_subspaces[1].samples)
    for subspace in samples_subspaces[2:end]
        start_ind, stop_ind = length(samples_subspaces[1].samples)+1, length(samples_subspaces[1].samples)+length(subspace.samples)
        subspace.info.samples_ind[1] = start_ind:stop_ind

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
    n::Integer,
    sampling_algorithm::A,
    integration_algorithm::I,
    sampling_kwargs::N
) where {N<:NamedTuple, A<:AbstractSamplingAlgorithm, I<:IntegrationAlgorithm}

    @info "Sampling subspace #$space_id"
    sampling_wc_start = Dates.Time(Dates.now())
    sampling_cpu_time = CPUTime.@CPUelapsed begin
        samples_subspace = bat_sample(posterior, sampling_algorithm).result
    end
    sampling_wc_stop = Dates.Time(Dates.now())

    integration_wc_start = Dates.Time(Dates.now())
    integration_cpu_time = CPUTime.@CPUelapsed begin
        integras_subspace = bat_integrate(samples_subspace, integration_algorithm).result
    end
    integration_wc_stop = Dates.Time(Dates.now())

    samples_subspace_reweighted = DensitySampleVector(
        (
            samples_subspace.v,
            samples_subspace.logd,
            integras_subspace.val .* samples_subspace.weight ./ sum(samples_subspace.weight),
            samples_subspace.info,
            samples_subspace.aux
        )
    )

    info_subspace = TypedTables.Table(
            density_integral = [integras_subspace],
            sampling_cpu_time = [sampling_cpu_time],
            integration_cpu_time = [integration_cpu_time],
            sampling_wc = [Dates.value(sampling_wc_start):Dates.value(sampling_wc_stop)],
            integration_wc = [Dates.value(integration_wc_start):Dates.value(integration_wc_stop)],
            worker_id = [Distributed.myid()],
            n_threads = [Threads.nthreads()],
            samples_ind = [0:0],
            sum_weights = [sum(samples_subspace.weight)],
        )

    return (samples = samples_subspace_reweighted, info = info_subspace)
end

function convert_to_posterior(posterior::PosteriorDensity, partition_tree::SpacePartTree; extend_bounds::Bool=true)

    if extend_bounds
        # Exploration samples might not always cover properly tails of the distribution.
        # We will extend boudnaries of the partition tree with original bounds which are:
        lo_bounds = getprior(posterior).bounds.vol.lo
        hi_bounds = getprior(posterior).bounds.vol.hi
        extend_tree_bounds!(partition_tree, lo_bounds, hi_bounds)
    end

    #Get flattened rectangular parameter bounds from tree
    subspaces_rect_bounds = get_tree_par_bounds(partition_tree)

    n_params = size(subspaces_rect_bounds[1])[1]

    posterior_array = map(
        x -> begin
            bounds = HyperRectBounds(x[:,1], x[:,2])
            prior_dist = BAT.truncate_density(getprior(posterior), bounds)
            PosteriorDensity(getlikelihood(posterior), prior_dist)
        end,
    subspaces_rect_bounds)

    return posterior_array
end
