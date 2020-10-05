# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct MCMCChainPoolInit <: MCMCInitAlgorithm

 MCMC chain pool initialization strategy.

Fields:
* `init_tries_per_chain`: Interval that specifies the minimum and maximum
  number of tries per MCMC chain to find a suitable starting position. Many
  candidate chains will be created and run for a short time. The chains with
  the best performance will be selected for tuning/burn-in and MCMC sampling
  run. Defaults to `IntervalSets.ClosedInterval(8, 128)`.
* `max_nsamples_init`: Maximum number of MCMC samples for each candidate
  chain. Defaults to 25. Definition of a sample depends on sampling algorithm.
* `max_nsteps_init`: Maximum number of MCMC steps for each candidate chain.
  Defaults to 250. Definition of a step depends on sampling algorithm.
* `max_time_init::Int`: Maximum wall-clock time to spend per candidate chain,
  in seconds. Defaults to `Inf`.
"""
@with_kw struct MCMCChainPoolInit <: MCMCInitAlgorithm
    init_tries_per_chain::ClosedInterval{Int64} = ClosedInterval(8, 128)
    max_nsamples_init::Int64 = 25
    max_nsteps_init::Int64 = 250
    max_time_init::Float64 = Inf
end

export MCMCChainPoolInit


##!!!!!!! missing: gen chain init values

_construct_chain(rngpart::RNGPartition, id::Integer, chainspec::MCMCSpec) =
    chainspec(AbstractRNG(rngpart, id), id)

_gen_chains(
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    chainspec::MCMCSpec,
) = [_construct_chain(rngpart, id, chainspec) for id in ids]


#!!!!! Change arg list
function mcmc_init!(
    rng::AbstractRNG,
    output::OptionalDensitySampleVector,
    algorithm::SomeAlgorithm,
    density::AbstractDensity,
    nchains::Int;
    tuner::MCMCTuningAlgorithm = MCMCTuningAlgorithm(chainspec.algorithm),
    init::MCMCInitAlgorithm = MCMCChainPoolInit(),
    callback::Function = nop_func
)
    @info "Trying to generate $nchains viable MCMC chain(s)."

    min_nviable::Int = minimum(init_strategy.init_tries_per_chain) * nchains
    max_ncandidates::Int = maximum(init_strategy.init_tries_per_chain) * nchains

    rngpart = RNGPartition(rng, Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    dummy_chain = MCMCIterator(algorithm, density, deepcopy(rng), 1)
    dummy_tuner = tuner_config(dummy_chain)

    chains = similar([dummy_chain], 0)
    tuners = similar([dummy_tuner], 0)
    cycle::Int = 1

    while length(tuners) < min_nviable && ncandidates < max_ncandidates
        n = min(min_nviable, max_ncandidates - ncandidates)
        @debug "Generating $n $(cycle > 1 ? "additional " : "")MCMC chain(s)."

        new_chains = _gen_chains(rngpart, ncandidates .+ (one(Int64):n), chainspec)

        filter!(isvalid, new_chains)

        new_tuners = tuner_config.(new_chains)
        tuning_init!.(new_tuners, new_chains)
        ncandidates += n

        @debug "Testing $(length(new_tuners)) MCMC chain(s)."

        mcmc_iterate!
            output, new_chains;
            max_nsamples = max(5, div(init_strategy.max_nsamples_init, 5)),
            max_nsteps =  max(50, div(init_strategy.max_nsteps_init, 5)),
            max_time = init_strategy.max_time_init / 5,
            callback = callback
        )

        viable_idxs = findall(isviable.(new_chains))
        viable_tuners = new_tuners[viable_idxs]
        viable_chains = new_chains[viable_idxs]

        @debug "Found $(length(viable_idxs)) viable MCMC chain(s)."

        if !isempty(viable_tuners)
            # ToDo: Use mcmc_iterate! instead of run_tuning_iterations! ?
            mcmc_iterate!(
                output, viable_chains;
                max_nsamples = init_strategy.max_nsamples_init,
                max_nsteps = init_strategy.max_nsteps_init,
                max_time = init_strategy.max_time_init,
                callback = callback
            )

            nsamples_thresh = floor(Int, 0.8 * median([nsamples(chain) for chain in viable_chains]))
            good_idxs = findall(chain -> nsamples(chain) >= nsamples_thresh, viable_chains)
            @debug "Found $(length(viable_tuners)) MCMC chain(s) with at least $(nsamples_thresh) samples."

            append!(chains, view(viable_chains, good_idxs))
            append!(tuners, view(viable_tuners, good_idxs))
        end

        cycle += 1
    end

    length(tuners) < min_nviable && error("Failed to generate $min_nviable viable MCMC chains")

    m = nchains
    tidxs = LinearIndices(tuners)
    n = length(tidxs)

    mode_1 = tuners[1].stats.mode
    modes = Array{eltype(mode_1)}(undef, length(mode_1), n)
    for i in tidxs
        modes[:,i] = tuners[i].stats.mode
    end

    final_chains = similar(chains, zero(Int))
    final_tuners = similar(tuners, zero(Int))

    if 2 <= m < size(modes, 2)
        clusters = kmeans(modes, m, init = KmCentralityAlg())
        clusters.converged || error("k-means clustering of MCMC chains did not converge")

        mincosts = fill(Inf, m)
        chain_sel_idxs = fill(0, m)

        for i in tidxs
            j = clusters.assignments[i]
            if clusters.costs[i] < mincosts[j]
                mincosts[j] = clusters.costs[i]
                chain_sel_idxs[j] = i
            end
        end

        @assert all(j -> j in tidxs, chain_sel_idxs)

        for i in sort(chain_sel_idxs)
            push!(final_chains, chains[i])
            push!(final_tuners, tuners[i])
        end
    else
        @assert length(chains) == nchains
        resize!(final_chains, nchains)
        copyto!(final_chains, chains)

        @assert length(tuners) == nchains
        resize!(final_tuners, nchains)
        copyto!(final_tuners, tuners)
    end


    @info "Selected $(length(final_tuners)) MCMC chain(s)."

    (chains = final_chains, tuners = final_tuners)
end
