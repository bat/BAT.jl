# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    PartitionedSampling

*BAT-internal, not part of stable public API.*

The algorithm that partitions parameter space by multiple subspaces and
samples/integrates them independently (See arXiv reference).

Constructor:

    PartitionedSampling(;kwargs...)

Optional Parameters/settings (`kwargs`):

    * `sampler::S = MetropolisHastings()` algorithm to generate samples.
    * `exploration_sampler::S = sampler` algorithm to generate exploration samples.
    * `partitioner::P = KDTreePartitioning()` algorithm to partition parameter space.
    * `integrator::I = AHMIntegration()` algorithm to integrate posterior.
    * `exploration_kwargs::NamedTuple = NamedTuple()` kwargs to be used in exploration sampler.
    * `sampling_kwargs::NamedTuple = NamedTuple()` kwargs to be used in subspace sampler.
    * `n_exploration::Tuple{Integer,Integer} = (10^2, 40)` number of exploration iterations.

"""
@with_kw struct PartitionedSampling{S<:AbstractSamplingAlgorithm, E<:AbstractSamplingAlgorithm,
    I<:IntegrationAlgorithm, P<:SpacePartitioningAlgorithm} <: AbstractSamplingAlgorithm
    sampler::S = MetropolisHastings()
    exploration_sampler::E = sampler
    partitioner::P = KDTreePartitioning()
    integrator::I = AHMIntegration()
    exploration_kwargs::NamedTuple = NamedTuple()
    sampling_kwargs::NamedTuple = NamedTuple()
    n_exploration::Tuple{Integer,Integer} = (10^2, 20)
end

export PartitionedSampling


"""
    function bat_sample(
        posterior::PosteriorDensity,
        n::Tuple{Integer,Integer, Integer},
        algorithm::PartitionedSampling;
    )

*BAT-internal, not part of stable public API.*

Generate samples from `posterior` using `PartitionedSampling()` algorithm (See arXiv reference).
`n` must be a tuple `(nsteps, nchains, npartitions)`.

Returns a NamedTuple of the shape

    ```julia
    (result = X::DensitySampleVector,
    info = Y::TypedTables.Table,
    exp_samples = Z::DensitySampleVector,
    part_tree = T::SpacePartTree,
    cost_values = A::AbstractArray,
    ...)
    ```
"""
function bat_sample(
        posterior::PosteriorDensity,
        n::Tuple{Integer,Integer, Integer},
        algorithm::PartitionedSampling
    )

    n_samples, n_chains, n_subspaces = n

    @info "Generating Exploration Samples"
    exploration_samples = bat_sample(posterior, algorithm.n_exploration, algorithm.exploration_sampler; algorithm.exploration_kwargs...).result

    @info "Constructing Partition Tree"
    partition_tree, cost_values = partition_space(exploration_samples, n_subspaces, algorithm.partitioner)
    # Convert 'partition_tree' structure into a set of truncated posteriors:
    posteriors_array = convert_to_posterior(posterior, partition_tree, extend_bounds = algorithm.partitioner.extend_bounds)

    @info "Sampling Subspaces"
    iterator_subspaces = [
        [subspace_ind, posteriors_array[subspace_ind],
        (n_samples, n_chains),
        algorithm.sampler,
        algorithm.integrator,
        algorithm.sampling_kwargs] for subspace_ind in Base.OneTo(n_subspaces)]
    samples_subspaces = pmap(inp -> sample_subspace(inp...), iterator_subspaces)

    @info "Combining Samples"

    # Save indices of a samples from different subspaces in a column
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
    n::Tuple{Integer,Integer},
    sampling_algorithm::A,
    integration_algorithm::I,
    sampling_kwargs::N
) where {N<:NamedTuple, A<:AbstractSamplingAlgorithm, I<:IntegrationAlgorithm}

    sampling_wc_start = Dates.Time(Dates.now())
    sampling_cpu_time = @CPUelapsed begin
        samples_subspace = bat_sample(posterior, n, sampling_algorithm; sampling_kwargs...).result
    end
    sampling_wc_stop = Dates.Time(Dates.now())

    integration_wc_start = Dates.Time(Dates.now())
    integration_cpu_time = @CPUelapsed begin
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
            samples_ind = [0:0] #will be specified when samples are merged
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

    #ToDo: Should be a better way to get n_params from posterior:
    n_params = size(subspaces_rect_bounds[1])[1]
    posterior_array = PosteriorDensity[]

    for subspace in subspaces_rect_bounds
        #ToDo: Use NamedTupleDist as a prior:
        bounds = BAT.HyperRectBounds(subspace[:,1], subspace[:,2],  repeat([BAT.hard_bounds], n_params))
        const_dens =  BAT.ConstDensity(bounds,0.0)
        push!(posterior_array, PosteriorDensity(getlikelihood(posterior), const_dens))
    end
    return posterior_array
end
