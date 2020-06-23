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
    partition_tree, cost_values = partition_space(exploration_samples, n_subspaces-1, algorithm.partitioner)
    posteriors_array = partitioned_priors(posterior, partition_tree)

    @info "Sample Parallel"
    #ToDo: Convert partition_tree -> set of posterior with corresponding bounded priors
    iterator_subspaces = [
        [subspace_ind, posteriors_array[subspace_ind],
        (n_samples, n_chains),
        algorithm.sampler,
        algorithm.integrator,
        algorithm.sampling_kwargs] for subspace_ind in Base.OneTo(n_subspaces)]

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

    sampling_wc_start = Dates.Time(Dates.now())
    sampling_cpu_time = @CPUelapsed begin
        samples_subspace = bat_sample(posterior, n, sampling_algorithm; sampling_kwargs...).result
    end
    sampling_wc_stop = Dates.Time(Dates.now())

    integration_cpu_time = @CPUelapsed begin
        integras_subspace = bat_integrate(samples_subspace, integration_algorithm).result
    end

    # α = exp(log(integras_subspace) + log(prior_normalization))
    # samples_subspace.weight .= α.val .* samples_subspace.weight ./ sum(samples_subspace.weight)

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
            sampling_time = [sampling_cpu_time],
            sampling_wc_start = [Dates.value(sampling_wc_start)],
            sampling_wc_stop = [Dates.value(sampling_wc_stop)],
            integration_time = [integration_cpu_time],
            worker_id = [Distributed.myid()],
            n_threads = [Threads.nthreads()],
            samples_ind = [10:30]
        )

    return (samples = samples_subspace_reweighted, info = info_subspace)
end

function partitioned_priors(posterior::PosteriorDensity, partition_tree::SpacePartTree)

    lo_bounds = getprior(posterior).bounds.vol.lo
    hi_bounds = getprior(posterior).bounds.vol.hi

    n_params = length(lo_bounds)

    subspaces_rect_bounds = get_tree_par_bounds(partition_tree)

    lo_tree_bounds = [minimum(hcat([tree_bound[:,1] for tree_bound in subspaces_rect_bounds]...), dims=2)...]
    hi_tree_bounds = [maximum(hcat([tree_bound[:,2] for tree_bound in subspaces_rect_bounds]...), dims=2)...]

    to_be_extended = [Pair.(lo_tree_bounds, lo_bounds); Pair.(hi_tree_bounds, hi_bounds)]

    posterior_array = PosteriorDensity[]

    for subspace in subspaces_rect_bounds
        bounds_tmp = replace(subspace, to_be_extended...)
        bounds = BAT.HyperRectBounds(bounds_tmp[:,1], bounds_tmp[:,2],  repeat([BAT.hard_bounds], n_params))
        const_dens =  BAT.ConstDensity(bounds,0.0)
        push!(posterior_array, PosteriorDensity(getlikelihood(posterior), const_dens))
    end
    return posterior_array
end
