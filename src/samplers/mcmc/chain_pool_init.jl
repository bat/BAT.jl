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
    algorithm::Union{TransformedMCMCSampling, MCMCAlgorithm}, # TODO: replace with MCMCAlgorithm, temporary during transformed transition
    m::BATMeasure,
    initval_alg::InitvalAlgorithm,
    parent_context::BATContext
)
    new_context = set_rng(parent_context, AbstractRNG(rngpart, id))
    v_init = bat_initval(m, initval_alg, new_context).result
    return algorithm isa TransformedMCMCSampling ? TransformedMCMCIterator(algorithm, m, id, v_init, new_context) : MCMCIterator(algorithm, m, id, v_init, new_context)
end

_gen_chains(
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    algorithm::Union{TransformedMCMCSampling, MCMCAlgorithm}, # TODO: replace with MCMCAlgorithm, temporary during transformed transition
    m::BATMeasure,
    initval_alg::InitvalAlgorithm,
    context::BATContext
) = [_construct_chain(rngpart, id, algorithm, m, initval_alg, context) for id in ids]

# TODO AC discuss
function _cluster_selection(
    chains::AbstractVector{<:MCMCIterator},
    tuners,
    outputs::AbstractVector{<:DensitySampleVector},
    scale::Real=3,
    decision_range_skip::Real=0.9,
)
    logds_by_chain = [view(s.logd,(floor(Int,decision_range_skip*length(s))):length(s)) for s in outputs]
    medians = [median(x) for x in logds_by_chain]
    stddevs = [std(x) for x in logds_by_chain]

    # yet uncategoriesed
    uncat = eachindex(chains, tuners, outputs, logds_by_chain, stddevs, medians)

    # clustered indices
    cidxs = Vector{Vector{eltype(uncat)}}()
    # categories all to clusters
    while length(uncat) > 0
        idxmin = findmin(view(stddevs,uncat))[2]

        cidx_sel = map(means_remaining_uncat -> abs(means_remaining_uncat-medians[uncat[idxmin]]) < scale*stddevs[uncat[idxmin]], view(medians,uncat))

        push!(cidxs, uncat[cidx_sel])
        uncat = uncat[.!cidx_sel]
    end
    medians_c = [ median(reduce(vcat, view(logds_by_chain, ids))) for ids in cidxs]
    idx_order = sortperm(medians_c, rev=true)

    chains_by_cluster = [ view(chains, ids) for ids in cidxs[idx_order]]
    tuners_by_cluster = [ view(tuners, ids) for ids in cidxs[idx_order]]
    outputs_by_cluster = [ view(outputs, ids) for ids in cidxs[idx_order]]
    ( chains = chains_by_cluster[1], tuners = tuners_by_cluster[1], outputs = outputs_by_cluster[1], )
end

function mcmc_init!(
    algorithm::MCMCAlgorithm, # TODO: resolve usage of MCMCAlgorithms
    m::BATMeasure,
    nchains::Integer,
    init_alg::MCMCChainPoolInit,
    tuning_alg::Union{MCMCTuningAlgorithm, MCMCTuningAlgorithm},
    nonzero_weights::Bool,
    callback::Function,
    context::BATContext
)::NamedTuple{(:chains, :tuners, :temperers, :outputs), Tuple{Vector, Vector, Vector, Vector}}

    sampling = TransformedMCMCSampling(
                tuning_alg = tuning_alg,
                proposal = _get_proposal(algorithm, m, context, bat_initval(m, init_alg.initval_alg, context).result), # TODO MD: Resolve initiation of proposal
                nchains = nchains,
                init = init_alg,
                nonzero_weights = nonzero_weights,
                callback = callback
    )

    mcmc_init!(sampling, m, context)
end

function mcmc_init!(
    sampling::TransformedMCMCSampling,
    m::BATMeasure,
    init::MCMCChainPoolInit,
    callback::Function,
    context::BATContext
)::NamedTuple{(:chains, :tuners, :temperers, :outputs), Tuple{Vector, Vector, Vector, Vector}} # 'Any' seems to be too general for type inference 
    
    @unpack nchains, tuning_alg, nonzero_weights = sampling

    @info "MCMCChainPoolInit: trying to generate $nchains viable MCMC chain(s)."

    initval_alg = init.initval_alg

    min_nviable::Int = minimum(init.init_tries_per_chain) * nchains
    max_ncandidates::Int = maximum(init.init_tries_per_chain) * nchains

    rngpart = RNGPartition(get_rng(context), Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    @debug "Generating dummy MCMC chain to determine chain, output and tuner types." #TODO: remove!

    dummy_context = deepcopy(context)
    dummy_initval = unshaped(bat_initval(m, InitFromTarget(), dummy_context).result, varshape(m))

    # TODO resolve, temporary workaround during transformed transition
    if sampling isa TransformedMCMCSampling
        dummy_chain = TransformedMCMCIterator(sampling, m, 1, dummy_initval, dummy_context) 
        dummy_tuner = get_tuner(tuning_alg, dummy_chain)
        dummy_temperer = get_temperer(sampling.tempering, m)
    else 
        dummy_chain = MCMCIterator(sampling, m, 1, dummy_initval, dummy_context)
        dummy_tuner = tuning_alg(dummy_chain)
        dummy_temperer = nothing
    end 

    chains = similar([dummy_chain], 0)
    tuners = similar([dummy_tuner], 0)
    temperers = similar([dummy_temperer], 0)
    outputs = similar([DensitySampleVector(dummy_chain)], 0)

    init_tries::Int = 1

    while length(tuners) < min_nviable && ncandidates < max_ncandidates
        viable_tuners = similar(tuners, 0)
        viable_chains = similar(chains, 0)
        viable_temperers = similar(temperers, 0)
        viable_outputs = similar(outputs, 0) #TODO

        # as the iteration after viable check is more costly, fill up to be at least capable to skip a complete reiteration.
        while length(viable_tuners) < min_nviable-length(tuners) && ncandidates < max_ncandidates
            n = min(min_nviable, max_ncandidates - ncandidates)
            @debug "Generating $n $(init_tries > 1 ? "additional " : "")candidate MCMC chain(s)."

            new_chains = _gen_chains(rngpart, ncandidates .+ (one(Int64):n), sampling, m, initval_alg, context)

            filter!(isvalidchain, new_chains)
            if sampling isa TransformedMCMCSampling # TODO: resolve, temporary workaround during transformed transition
                new_tuners = get_tuner.(Ref(tuning_alg), new_chains)
                new_temperers = fill(get_temperer(sampling.tempering, m), size(new_tuners,1))
            else
                new_tuners = tuning_alg.(new_chains)
                new_outputs = DensitySampleVector.(new_chains)
            end 

            next_cycle!.(new_chains)
            tuning_init!.(new_tuners, new_chains, init.nsteps_init)
            ncandidates += n

            @debug "Testing $(length(new_chains)) candidate MCMC chain(s)."
            if sampling isa TransformedMCMCSampling # TODO: resolve, temporary workaround during transformed transition
                transformed_mcmc_iterate!(
                    new_chains, new_tuners, new_temperers,
                    max_nsteps = clamp(div(init.nsteps_init, 5), 10, 50),
                    callback = callback,
                    nonzero_weights = nonzero_weights
                )
                new_outputs = getproperty.(new_chains, :samples) #TODO ?
                global gstate_iterator = (new_chains, new_outputs, new_tuners, new_temperers, viable_outputs)
            else
                mcmc_iterate!(
                    new_outputs, new_chains, new_tuners;
                    max_nsteps = clamp(div(init.nsteps_init, 5), 10, 50),
                    callback = callback,
                    nonzero_weights = nonzero_weights
                )
            end
            # testing if chains are viable:
            viable_idxs = findall(isviablechain.(new_chains))
            @info length.(new_outputs)

            append!(viable_tuners, new_tuners[viable_idxs])
            append!(viable_chains, new_chains[viable_idxs])
            append!(viable_outputs, new_outputs[viable_idxs])
            if sampling isa TransformedMCMCSampling
                append!(viable_temperers, new_temperers[viable_idxs])
            end
        end

        @debug "Found $(length(viable_tuners)) viable MCMC chain(s)."

        if !isempty(viable_chains)
            desc_string = string("Init try ", init_tries, " for nvalid=", length(viable_tuners), " of min_nviable=", length(tuners), "/", min_nviable )
            progress_meter = ProgressMeter.Progress(length(viable_tuners) * init.nsteps_init, desc=desc_string, barlen=80-length(desc_string), dt=0.1)
            
            if sampling isa TransformedMCMCSampling
                transformed_mcmc_iterate!(
                    viable_chains, viable_tuners, viable_temperers;
                    max_nsteps = init.nsteps_init,
                    callback = (kwargs...)-> let pm=progress_meter; ProgressMeter.next!(pm) ; end,
                    nonzero_weights = nonzero_weights
                )
            else
                mcmc_iterate!(
                    viable_outputs, viable_chains, viable_tuners;
                    max_nsteps = init.nsteps_init,
                    callback = (kwargs...)-> let pm=progress_meter, callback=callback ; callback(kwargs) ; ProgressMeter.next!(pm) ; end,
                    nonzero_weights = nonzero_weights
                )
            end

            
            ProgressMeter.finish!(progress_meter)

            nsamples_thresh = floor(Int, 0.8 * median([nsamples(chain) for chain in viable_chains]))
            good_idxs = findall(chain -> nsamples(chain) >= nsamples_thresh, viable_chains)
            @debug "Found $(length(viable_chains)) MCMC chain(s) with at least $(nsamples_thresh) unique accepted samples."

            append!(chains, view(viable_chains, good_idxs))
            append!(tuners, view(viable_tuners, good_idxs))
            if sampling isa TransformedMCMCSampling
                append!(temperers, view(viable_temperers, good_idxs))
                append!(outputs, view(viable_outputs, good_idxs))
            else 
                append!(outputs, view(viable_outputs, good_idxs))
            end
        end

        init_tries += 1
    end
    
    # Disabled, as it kept causing issues with too few viable chains
    # # TODO AC
    # if true
    #     @unpack chains, tuners, outputs = _cluster_selection(chains, tuners, outputs, 15) # default scale for _cluster_selection() seems to be too strict. Relaxed it to 15
    # else
    #     length(tuners) < min_nviable && error("Failed to generate $min_nviable viable MCMC chains")
    # end

    length(tuners) < min_nviable && error("Failed to generate $min_nviable viable MCMC chains")


    m = nchains
    tidxs = LinearIndices(chains)
    n = length(tidxs)

    modes = hcat(broadcast(samples -> Array(bat_findmode(samples, MaxDensitySearch(), context).result), outputs)...)

    final_chains = similar(chains, 0)
    final_tuners = similar(tuners, 0)
    final_temperers = similar(temperers, 0)
    final_outputs = similar(outputs, 0)


    # TODO: should we put this into a function?
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
            if sampling isa TransformedMCMCSampling
                push!(final_temperers, temperers[i])
            end
            push!(final_outputs, outputs[i])
        end
    elseif m == 1
        i = findmax(nsamples.(chains))[2]
        push!(final_chains, chains[i])
        push!(final_tuners, tuners[i])
        if sampling isa TransformedMCMCSampling
            push!(final_temperers, temperers[i])
        end
        push!(final_outputs, outputs[i])
    else
        println("$(length(chains)) == $nchains")
        @assert length(chains) == nchains
        resize!(final_chains, nchains)
        copyto!(final_chains, chains)

        @assert length(tuners) == nchains
        resize!(final_tuners, nchains)
        copyto!(final_tuners, tuners)

        if sampling isa TransformedMCMCSampling
            @assert length(temperers) == nchains
            resize!(final_temperers, nchains)
            copyto!(final_temperers, temperers)
        end

        @assert length(outputs) == nchains
        resize!(final_outputs, nchains)
        copyto!(final_outputs, outputs)
    end

    @info "Selected $(length(final_chains)) MCMC chain(s)."
    tuning_postinit!.(final_tuners, final_chains, final_outputs) #TODO: implement

    global gstate_post_iteration_init = (final_chains, final_tuners, final_temperers, final_outputs)

    (chains = final_chains, tuners = final_tuners, temperers = final_temperers, outputs = final_outputs)
end
