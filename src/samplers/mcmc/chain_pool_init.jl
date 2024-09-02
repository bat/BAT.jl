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



function _construct_state(
    sampling::MCMCSampling,
    target::BATMeasure,
    rngpart::RNGPartition,
    id::Integer,
    parent_context::BATContext
)
    new_context = set_rng(parent_context, AbstractRNG(rngpart, id))
    v_init = bat_initval(target, initval_alg, new_context).result
    return MCMCState(sampling, target, id, v_init, new_context)
end

_gen_states(
    sampling::MCMCSampling,
    target::BATMeasure,
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    context::BATContext
) = [_construct_state(sampling, target, rngpart, id, context) for id in ids]


function mcmc_init!(
    sampling::MCMCSampling,
    target::BATMeasure,
    init_alg::MCMCChainPoolInit,
    callback::Function,
    context::BATContext
)
    
    @unpack nchains, tuning_alg, nonzero_weights = sampling

    @info "MCMCChainPoolInit: trying to generate $nchains viable MCMC chain(s)."

    initval_alg = init_alg.initval_alg

    min_nviable::Int = minimum(init_alg.init_tries_per_chain) * nchains
    max_ncandidates::Int = maximum(init_alg.init_tries_per_chain) * nchains

    rngpart = RNGPartition(get_rng(context), Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    @debug "Generating dummy MCMC chain to determine chain, output and tuner types."

    dummy_context = deepcopy(context)
    dummy_initval = unshaped(bat_initval(target, InitFromTarget(), dummy_context).result, varshape(target))
    dummy_state = MCMCState(sampling, target, 1, dummy_initval, dummy_context)
    dummy_tuner = tuning_alg(dummy_state)

    states = similar([dummy_state], 0)
    tuners = similar([dummy_tuner], 0)
    outputs = similar([DensitySampleVector(dummy_state)], 0)
    cycle::Int32 = 1

    while length(tuners) < min_nviable && ncandidates < max_ncandidates
        n = min(min_nviable, max_ncandidates - ncandidates)
        @debug "Generating $n $(cycle > 1 ? "additional " : "")candidate MCMC state(s)."

        new_states = _gen_states(sampling, target, rngpart, ncandidates .+ (one(Int64):n), context)

        filter!(isvalidstate, new_states)

        new_tuners = tuning_alg.(new_states)
        new_outputs = DensitySampleVector.(new_states)
        next_cycle!.(new_states)
        tuning_init!.(new_tuners, new_states, init_alg.nsteps_init)
        ncandidates += n

        @debug "Testing $(length(new_tuners)) candidate MCMC state(s)."

        mcmc_iterate!(
            new_outputs, new_states, new_tuners;
            max_nsteps = clamp(div(init_alg.nsteps_init, 5), 10, 50),
            callback = callback,
            nonzero_weights = nonzero_weights
        )

        viable_idxs = findall(isviablestate.(new_states))
        viable_tuners = new_tuners[viable_idxs]
        viable_states = new_states[viable_idxs]
        viable_outputs = new_outputs[viable_idxs]

        @debug "Found $(length(viable_idxs)) viable MCMC state(s)."

        if !isempty(viable_tuners)
            mcmc_iterate!(
                viable_outputs, viable_states, viable_tuners;
                max_nsteps = init_alg.nsteps_init,
                callback = callback,
                nonzero_weights = nonzero_weights
            )

            nsamples_thresh = floor(Int, 0.8 * median([nsamples(state) for state in viable_states]))
            good_idxs = findall(state -> nsamples(state) >= nsamples_thresh, viable_states)
            @debug "Found $(length(viable_tuners)) MCMC state(s) with at least $(nsamples_thresh) unique accepted samples."

            append!(states, view(viable_states, good_idxs))
            append!(tuners, view(viable_tuners, good_idxs))
            append!(outputs, view(viable_outputs, good_idxs))
        end

        cycle += 1
    end

    length(tuners) < min_nviable && error("Failed to generate $min_nviable viable MCMC states")

    m = nstates
    tidxs = LinearIndices(tuners)
    n = length(tidxs)

    modes = hcat(broadcast(samples -> Array(bat_findmode(samples, MaxDensitySearch(), context).result), outputs)...)

    final_states = similar(states, 0)
    final_tuners = similar(tuners, 0)
    final_outputs = similar(outputs, 0)

    if 2 <= m < size(modes, 2)
        clusters = kmeans(modes, m, init = KmCentralityAlg())
        clusters.converged || error("k-means clustering of MCMC states did not converge")

        mincosts = fill(Inf, m)
        state_sel_idxs = fill(0, m)

        for i in tidxs
            j = clusters.assignments[i]
            if clusters.costs[i] < mincosts[j]
                mincosts[j] = clusters.costs[i]
                state_sel_idxs[j] = i
            end
        end

        @assert all(j -> j in tidxs, state_sel_idxs)

        for i in sort(state_sel_idxs)
            push!(final_states, states[i])
            push!(final_tuners, tuners[i])
            push!(final_outputs, outputs[i])
        end
    elseif m == 1
        i = findmax(nsamples.(states))[2]
        push!(final_states, states[i])
        push!(final_tuners, tuners[i])
        push!(final_outputs, outputs[i])
    else
        @assert length(states) == nstates
        resize!(final_states, nstates)
        copyto!(final_states, states)

        @assert length(tuners) == nstates
        resize!(final_tuners, nstates)
        copyto!(final_tuners, tuners)

        @assert length(outputs) == nstates
        resize!(final_outputs, nstates)
        copyto!(final_outputs, outputs)
    end

    @info "Selected $(length(final_tuners)) MCMC state(s)."
    tuning_postinit!.(final_tuners, final_states, final_outputs)

    (states = final_states, tuners = final_tuners, outputs = final_outputs)
end
