# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct MCMCChainPoolInit <: MCMCInitAlgorithm

MCMC chain pool initialization strategy.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MCMCChainPoolInit <: MCMCInitAlgorithm
    init_tries_per_chain::ClosedInterval{Int64} = ClosedInterval(8, 128)
    nsteps_init::Int64 = 1000
    initval_alg::InitvalAlgorithm = InitFromTarget()
end

export MCMCChainPoolInit

apply_trafo_to_init(trafo::Function, initalg::MCMCChainPoolInit) = MCMCChainPoolInit(
    initalg.init_tries_per_chain,
    initalg.nsteps_init,
    apply_trafo_to_init(trafo, initalg.initval_alg)
)



function apply_trafo_to_init(trafo::Function, initalg::MCMCChainPoolInit)
    MCMCChainPoolInit(
    initalg.init_tries_per_chain,
    initalg.nsteps_init,
    apply_trafo_to_init(trafo, initalg.initval_alg)
    )
end



function _construct_chain(
    rngpart::RNGPartition,
    id::Integer,
    algorithm::MCMCAlgorithm,
    density::AbstractMeasureOrDensity,
    initval_alg::InitvalAlgorithm
)
    rng = AbstractRNG(rngpart, id)
    v_init = bat_initval(rng, density, initval_alg).result

    MCMCIterator(rng, algorithm, density, id, v_init)
end

_gen_chains(
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    algorithm::MCMCAlgorithm,
    density::AbstractMeasureOrDensity,
    initval_alg::InitvalAlgorithm
) = [_construct_chain(rngpart, id, algorithm, density, initval_alg) for id in ids]


function mcmc_init!(
    rng::AbstractRNG,
    algorithm::MCMCAlgorithm,
    density::AbstractMeasureOrDensity,
    nchains::Integer,
    init_alg::MCMCChainPoolInit,
    tuning_alg::MCMCTuningAlgorithm,
    nonzero_weights::Bool,
    callback::Function
)
    @info "MCMCChainPoolInit: trying to generate $nchains viable MCMC chain(s)."

    initval_alg = init_alg.initval_alg

    min_nviable::Int = minimum(init_alg.init_tries_per_chain) * nchains
    max_ncandidates::Int = maximum(init_alg.init_tries_per_chain) * nchains

    rngpart = RNGPartition(rng, Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    @debug "Generating dummy MCMC chain to determine chain, output and tuner types."

    dummy_initval = unshaped(bat_initval(rng, density, InitFromTarget()).result, varshape(density))
    dummy_chain = MCMCIterator(deepcopy(rng), algorithm, density, 1, dummy_initval)
    dummy_tuner = tuning_alg(dummy_chain)

    chains = similar([dummy_chain], 0)
    tuners = similar([dummy_tuner], 0)
    outputs = similar([DensitySampleVector(dummy_chain)], 0)
    cycle::Int = 1

    while length(tuners) < min_nviable && ncandidates < max_ncandidates
        n = min(min_nviable, max_ncandidates - ncandidates)
        @debug "Generating $n $(cycle > 1 ? "additional " : "")candidate MCMC chain(s)."

        new_chains = _gen_chains(rngpart, ncandidates .+ (one(Int64):n), algorithm, density, initval_alg)

        filter!(isvalidchain, new_chains)

        new_tuners = tuning_alg.(new_chains)
        new_outputs = DensitySampleVector.(new_chains)
        next_cycle!.(new_chains)
        tuning_init!.(new_tuners, new_chains, init_alg.nsteps_init)
        ncandidates += n

        @debug "Testing $(length(new_tuners)) candidate MCMC chain(s)."

        mcmc_iterate!(
            new_outputs, new_chains, new_tuners;
            max_nsteps = clamp(div(init_alg.nsteps_init, 5), 10, 50),
            callback = callback,
            nonzero_weights = nonzero_weights
        )

        viable_idxs = findall(isviablechain.(new_chains))
        viable_tuners = new_tuners[viable_idxs]
        viable_chains = new_chains[viable_idxs]
        viable_outputs = new_outputs[viable_idxs]

        @debug "Found $(length(viable_idxs)) viable MCMC chain(s)."

        if !isempty(viable_tuners)
            mcmc_iterate!(
                viable_outputs, viable_chains, viable_tuners;
                max_nsteps = init_alg.nsteps_init,
                callback = callback,
                nonzero_weights = nonzero_weights
            )

            nsamples_thresh = floor(Int, 0.8 * median([nsamples(chain) for chain in viable_chains]))
            good_idxs = findall(chain -> nsamples(chain) >= nsamples_thresh, viable_chains)
            @debug "Found $(length(viable_tuners)) MCMC chain(s) with at least $(nsamples_thresh) unique accepted samples."

            append!(chains, view(viable_chains, good_idxs))
            append!(tuners, view(viable_tuners, good_idxs))
            append!(outputs, view(viable_outputs, good_idxs))
        end

        cycle += 1
    end

    length(tuners) < min_nviable && error("Failed to generate $min_nviable viable MCMC chains")

    m = nchains
    tidxs = LinearIndices(tuners)
    n = length(tidxs)

    modes = hcat(broadcast(samples -> Array(bat_findmode(rng, samples, MaxDensitySearch()).result), outputs)...)

    final_chains = similar(chains, 0)
    final_tuners = similar(tuners, 0)
    final_outputs = similar(outputs, 0)

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
            push!(final_outputs, outputs[i])
        end
    elseif m == 1
        i = findmax(nsamples.(chains))[2]
        push!(final_chains, chains[i])
        push!(final_tuners, tuners[i])
        push!(final_outputs, outputs[i])
    else
        @assert length(chains) == nchains
        resize!(final_chains, nchains)
        copyto!(final_chains, chains)

        @assert length(tuners) == nchains
        resize!(final_tuners, nchains)
        copyto!(final_tuners, tuners)

        @assert length(outputs) == nchains
        resize!(final_outputs, nchains)
        copyto!(final_outputs, outputs)
    end

    @info "Selected $(length(final_tuners)) MCMC chain(s)."
    tuning_postinit!.(final_tuners, final_chains, final_outputs)

    (chains = final_chains, tuners = final_tuners, outputs = final_outputs)
end
