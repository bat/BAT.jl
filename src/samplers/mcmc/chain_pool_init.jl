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


function _construct_chain_state(
    sampling::MCMCSampling,
    target::BATMeasure,
    rngpart::RNGPartition,
    id::Integer,
    initval_alg::InitvalAlgorithm,
    parent_context::BATContext
)
    new_context = set_rng(parent_context, AbstractRNG(rngpart, id))
    v_init = bat_initval(target, initval_alg, new_context).result
    return MCMCState(sampling, target, id, v_init, new_context)
end

_gen_chain_states(
    sampling::MCMCSampling,
    target::BATMeasure,
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    initval_alg::InitvalAlgorithm,
    context::BATContext
) = [_construct_chain_state(sampling, target, rngpart, id, initval_alg, context) for id in ids]


function mcmc_init!(
    sampling::MCMCSampling,
    target::BATMeasure,
    init_alg::MCMCChainPoolInit,
    callback::Function,
    context::BATContext
)::NamedTuple{(:chain_states, :tuners, :outputs, :temperers), Tuple{Vector{MCMCState}, Vector{AbstractMCMCTunerInstance}, Vector{DensitySampleVector}, Vector{AbstractMCMCTemperingInstance}}}
    @unpack tempering, nchains, tuning, nonzero_weights = sampling

    @info "MCMCChainPoolInit: trying to generate $nchains viable MCMC chain state(s)."

    initval_alg = init_alg.initval_alg

    min_nviable::Int = minimum(init_alg.init_tries_per_chain) * nchains
    max_ncandidates::Int = maximum(init_alg.init_tries_per_chain) * nchains

    rngpart = RNGPartition(get_rng(context), Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    @debug "Generating dummy MCMC chain state to determine chain state, output and tuner types."

    dummy_context = deepcopy(context)
    dummy_initval = unshaped(bat_initval(target, InitFromTarget(), dummy_context).result, varshape(target))
    dummy_mc_state = MCMCState(sampling, target, 1, dummy_initval, dummy_context)
    dummy_tuner = tuning(dummy_mc_state)
    dummy_temperer = create_temperering_state(tempering, target)


    mc_states = similar([dummy_mc_state], 0)
    tuners = similar([dummy_tuner], 0)
    outputs = similar([DensitySampleVector(dummy_mc_state)], 0)
    temperers = similar([dummy_temperer], 0)

    cycle::Int32 = 1

    while length(tuners) < min_nviable && ncandidates < max_ncandidates
        n = min(min_nviable, max_ncandidates - ncandidates)
        @debug "Generating $n $(cycle > 1 ? "additional " : "")candidate MCMC chain state(s)."

        new_mc_states = _gen_chain_states(sampling, target, rngpart, ncandidates .+ (one(Int64):n), initval_alg, context)

        filter!(isvalidstate, new_mc_states)

        new_tuners = tuning.(new_mc_states)
        new_outputs = DensitySampleVector.(new_mc_states)
        new_temperers = fill(create_temperering_state(tempering, target), size(new_tuners,1))
        next_cycle!.(new_mc_states)
        tuning_init!.(new_tuners, new_mc_states, init_alg.nsteps_init)
        ncandidates += n

        @debug "Testing $(length(new_tuners)) candidate MCMC chain state(s)."
        
        new_mc_states, new_tuners, new_temperers = mcmc_iterate!!(
            new_outputs, new_mc_states; 
            tuners = new_tuners, temperers = new_temperers,
            max_nsteps = clamp(div(init_alg.nsteps_init, 5), 10, 50),
            nonzero_weights = nonzero_weights
            )
        
            
        viable_idxs = findall(isviablestate.(new_mc_states))
        viable_tuners = new_tuners[viable_idxs]
        viable_mc_states = new_mc_states[viable_idxs]
        viable_outputs = new_outputs[viable_idxs]
        viable_temperers = new_temperers[viable_idxs]

        @debug "Found $(length(viable_idxs)) viable MCMC chain state(s)."

        if !isempty(viable_tuners)
            viable_mc_states, viable_tuners, viable_temperers = mcmc_iterate!!(
                viable_outputs, viable_mc_states; 
                tuners = viable_tuners, temperers = viable_temperers,
                max_nsteps = init_alg.nsteps_init,
                nonzero_weights = nonzero_weights
            )

            nsamples_thresh = floor(Int, 0.8 * median([nsamples(mc_state) for mc_state in viable_mc_states]))
            good_idxs = findall(mc_state -> nsamples(mc_state) >= nsamples_thresh, viable_mc_states)
            @debug "Found $(length(viable_tuners)) MCMC chain state(s) with at least $(nsamples_thresh) unique accepted samples."

            append!(mc_states, view(viable_mc_states, good_idxs))
            append!(tuners, view(viable_tuners, good_idxs))
            append!(outputs, view(viable_outputs, good_idxs))
            append!(temperers, view(viable_temperers, good_idxs))
        end

        cycle += 1
    end

    length(tuners) < min_nviable && error("Failed to generate $min_nviable viable MCMC chain states")

    m = nchains
    tidxs = LinearIndices(tuners)
    n = length(tidxs)

    modes = hcat(broadcast(samples -> Array(bat_findmode(samples, MaxDensitySearch(), context).result), outputs)...)

    final_mc_states = similar(mc_states, 0)
    final_tuners = similar(tuners, 0)
    final_outputs = similar(outputs, 0)
    final_temperers = similar(temperers, 0)

    if 2 <= m < size(modes, 2)
        clusters = kmeans(modes, m, init = KmCentralityAlg())
        clusters.converged || error("k-means clustering of MCMC chain states did not converge")

        mincosts = fill(Inf, m)
        mc_state_sel_idxs = fill(0, m)

        for i in tidxs
            j = clusters.assignments[i]
            if clusters.costs[i] < mincosts[j]
                mincosts[j] = clusters.costs[i]
                mc_state_sel_idxs[j] = i
            end
        end

        @assert all(j -> j in tidxs, mc_state_sel_idxs)

        for i in sort(mc_state_sel_idxs)
            push!(final_mc_states, mc_states[i])
            push!(final_tuners, tuners[i])
            push!(final_outputs, outputs[i])
            push!(final_temperers, temperers[i])
        end
    elseif m == 1
        i = findmax(nsamples.(mc_states))[2]
        push!(final_mc_states, mc_states[i])
        push!(final_tuners, tuners[i])
        push!(final_outputs, outputs[i])
        push!(final_temperers, temperers[i])
    else
        @assert length(mc_states) == n_mc_states
        resize!(final_mc_states, n_mc_states)
        copyto!(final_mc_states, mc_states)

        @assert length(tuners) == n_mc_states
        resize!(final_tuners, n_mc_states)
        copyto!(final_tuners, tuners)

        @assert length(outputs) == n_mc_states
        resize!(final_outputs, n_mc_states)
        copyto!(final_outputs, outputs)

        @assert length(temperers) == n_mc_states
        resize!(final_temperers, n_mc_states)
        copyto!(final_temperers, temperers)
    end

    @info "Selected $(length(final_tuners)) MCMC chain state(s)."
    tuning_postinit!.(final_tuners, final_mc_states, final_outputs)

    (chain_states = final_mc_states, tuners = final_tuners, outputs = final_outputs, temperers = final_temperers)
end
