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


function apply_trafo_to_init(f_transform::Function, initalg::MCMCChainPoolInit)
    MCMCChainPoolInit(
    initalg.init_tries_per_chain,
    initalg.nsteps_init,
    apply_trafo_to_init(f_transform, initalg.initval_alg)
    )
end


function _construct_mcmc_state(
    samplingalg::TransformedMCMC,
    target::BATMeasure,
    rngpart::RNGPartition,
    id::Integer,
    initval_alg::InitvalAlgorithm,
    parent_context::BATContext
)
    new_context = set_rng(parent_context, AbstractRNG(rngpart, id))
    v_init = bat_initval(target, initval_alg, new_context).result
    return MCMCState(samplingalg, target, Int32(id), v_init, new_context)
end

_gen_mcmc_states(
    samplingalg::TransformedMCMC,
    target::BATMeasure,
    rngpart::RNGPartition,
    ids::AbstractRange{<:Integer},
    initval_alg::InitvalAlgorithm,
    context::BATContext
) = [_construct_mcmc_state(samplingalg, target, rngpart, id, initval_alg, context) for id in ids]


function mcmc_init!(
    samplingalg::TransformedMCMC,
    target::BATMeasure,
    init_alg::MCMCChainPoolInit,
    callback::Function,
    context::BATContext
)::NamedTuple{(:mcmc_states, :outputs), Tuple{Vector{MCMCState}, Vector{DensitySampleVector}}}
    @unpack tempering, nchains, transform_tuning, proposal_tuning, nonzero_weights = samplingalg

    @info "MCMCChainPoolInit: trying to generate $nchains viable MCMC chain state(s)."

    initval_alg = init_alg.initval_alg

    min_nviable::Int = minimum(init_alg.init_tries_per_chain) * nchains
    max_ncandidates::Int = maximum(init_alg.init_tries_per_chain) * nchains

    rngpart = RNGPartition(get_rng(context), Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    @debug "Generating dummy MCMC chain state to determine chain state, output and tuner types."

    dummy_context = deepcopy(context)
    dummy_initval = unshaped(bat_initval(target, InitFromTarget(), dummy_context).result, varshape(target))

    dummy_mcmc_state = MCMCState(samplingalg, target, one(Int32), dummy_initval, dummy_context)

    mcmc_states = similar([dummy_mcmc_state], 0)
    outputs = similar([DensitySampleVector(dummy_mcmc_state)], 0)

    cycle::Int32 = 1

    while length(mcmc_states) < min_nviable && ncandidates < max_ncandidates
        n = min(min_nviable, max_ncandidates - ncandidates)
        @debug "Generating $n $(cycle > 1 ? "additional " : "")candidate MCMC chain state(s)."

        new_mcmc_states = _gen_mcmc_states(samplingalg, target, rngpart, ncandidates .+ (one(Int64):n), initval_alg, context)

        filter!(isvalidstate, new_mcmc_states)

        new_outputs = DensitySampleVector.(new_mcmc_states)

        next_cycle!.(new_mcmc_states)
        mcmc_tuning_init!!.(new_mcmc_states, init_alg.nsteps_init)
        new_mcmc_states = mcmc_update_z_position!!.(new_mcmc_states)
        ncandidates += n

        @debug "Testing $(length(new_mcmc_states)) candidate MCMC chain state(s)."
        
        new_mcmc_states = mcmc_iterate!!(
            new_outputs, new_mcmc_states;
            max_nsteps = clamp(div(init_alg.nsteps_init, 5), 10, 50),
            nonzero_weights = nonzero_weights
        )
        
        viable_idxs = findall(isviablestate.(new_mcmc_states))
        viable_mcmc_states = new_mcmc_states[viable_idxs]
        viable_outputs = new_outputs[viable_idxs]

        @debug "Found $(length(viable_idxs)) viable MCMC chain state(s)."

        if !isempty(viable_mcmc_states)
            viable_mcmc_states = mcmc_iterate!!(
                viable_outputs, viable_mcmc_states;
                max_nsteps = init_alg.nsteps_init,
                nonzero_weights = nonzero_weights
            )

            nsamples_thresh = floor(Int, 0.8 * median([nsamples(states) for states in viable_mcmc_states]))
            good_idxs = findall(states -> nsamples(states) >= nsamples_thresh, viable_mcmc_states)
            @debug "Found $(length(viable_mcmc_states)) MCMC chain state(s) with at least $(nsamples_thresh) unique accepted samples."

            append!(mcmc_states, view(viable_mcmc_states, good_idxs))
            append!(outputs, view(viable_outputs, good_idxs))
        end

        cycle += 1
    end

    length(mcmc_states) < min_nviable && error("Failed to generate $min_nviable viable MCMC chain states")

    m = nchains
    tidxs = LinearIndices(mcmc_states)
    n = length(tidxs)

    modes = hcat(broadcast(samples -> Array(bat_findmode(samples, MaxDensitySearch(), context).result), outputs)...)

    final_mcmc_states = similar(mcmc_states, 0)
    final_outputs = similar(outputs, 0)

    if 2 <= m < size(modes, 2)
        clusters = kmeans(modes, m, init = KmCentralityAlg())
        clusters.converged || error("k-means clustering of MCMC chain states did not converge")

        mincosts = fill(Inf, m)
        mcmc_states_sel_idxs = fill(0, m)

        for i in tidxs
            j = clusters.assignments[i]
            if clusters.costs[i] < mincosts[j]
                mincosts[j] = clusters.costs[i]
                mcmc_states_sel_idxs[j] = i
            end
        end

        @assert all(j -> j in tidxs, mcmc_states_sel_idxs)

        for i in sort(mcmc_states_sel_idxs)
            push!(final_mcmc_states, mcmc_states[i])
            push!(final_outputs, outputs[i])
        end
    elseif m == 1
        i = findmax(nsamples.(mcmc_states))[2]
        push!(final_mcmc_states, mcmc_states[i])
        push!(final_outputs, outputs[i])
    else
        @assert length(mcmc_states) == n_mc_states
        resize!(final_mcmc_states, n_mc_states)
        copyto!(final_mcmc_states, mcmc_states)

        @assert length(outputs) == n_mc_states
        resize!(final_outputs, n_mc_states)
        copyto!(final_outputs, outputs)
    end

    @info "Selected $(length(final_mcmc_states)) MCMC chain state(s)."
    mcmc_tuning_postinit!!.(final_mcmc_states, final_outputs)

    (mcmc_states = final_mcmc_states, outputs = final_outputs)
end
