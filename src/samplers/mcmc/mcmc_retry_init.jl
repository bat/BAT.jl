# This file is a part of BAT.jl, licensed under the MIT License (MIT).



# Draw a random init point for each walker for each chain
# And let the chains run for nsteps_init steps and unviable walkers get a new random position and let their chains run until 
# Leave the chains with viable walkers as is 
# strict lets the init fail if nothing moves

"""
    struct MCMCRetryInit <: MCMCInitAlgorithm

TODO

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MCMCRetryInit <: MCMCInitAlgorithm
    init_tries_per_chain::Int64 = 16
    nsteps_init::Int64 = 20
    initval_alg::InitvalAlgorithm = InitFromTarget()
    strict::Bool = true
end

export MCMCRetryInit

# TODO: MD, could be generalized for different initalgs?
function apply_trafo_to_init(f_transform::Function, initalg::MCMCRetryInit)
    MCMCRetryInit(
    initalg.init_tries_per_chain,
    initalg.nsteps_init,
    apply_trafo_to_init(f_transform, initalg.initval_alg),
    initalg.strict
    )
end


function mcmc_init!(
    samplingalg::TransformedMCMC,
    target::BATMeasure,
    init_alg::MCMCRetryInit,
    callback::Function,
    context::BATContext
)::NamedTuple{(:mcmc_states, :outputs), Tuple{Vector{MCMCState}, Vector{Vector{DensitySampleVector}}}}

    (;nchains, nonzero_weights) = samplingalg

    @info "MCMCChainPoolInit: trying to generate $nchains viable MCMC chain state(s)."

    initval_alg = init_alg.initval_alg

    min_nviable::Int = minimum(init_alg.init_tries_per_chain) * nchains
    max_ncandidates::Int = maximum(init_alg.init_tries_per_chain) * nchains

    rngpart = RNGPartition(get_rng(context), Base.OneTo(max_ncandidates))

    ncandidates::Int = 0

    @debug "Generating dummy MCMC chain state to determine chain state, output and tuner types."

    dummy_context = deepcopy(context)
    dummy_initval = [unshaped(bat_initval(target, InitFromTarget(), dummy_context).result, varshape(target))] # TODO, MD: expand to multiple walkers 

    dummy_mcmc_state = MCMCState(samplingalg, target, one(Int32), dummy_initval, dummy_context)

    mcmc_states = similar([dummy_mcmc_state], 0)
    outputs = similar([_empty_chain_outputs(dummy_mcmc_state)], 0)

    cycle::Int32 = 1

    while length(mcmc_states) < min_nviable && ncandidates < max_ncandidates
        n = min(min_nviable, max_ncandidates - ncandidates)
        @debug "Generating $n $(cycle > 1 ? "additional " : "")candidate MCMC chain state(s)."

        new_mcmc_states = _gen_mcmc_states(samplingalg, target, rngpart, ncandidates .+ (one(Int64):n), initval_alg, context)

        filter!(isvalidstate, new_mcmc_states)

        new_outputs = _empty_chain_outputs.(new_mcmc_states)

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

    init_alg.strict && length(mcmc_states) < min_nviable && error("Failed to generate $min_nviable viable MCMC chain states")

    @info "Selected $(length(mcmc_states)) MCMC chain state(s)."
    
    mcmc_tuning_postinit!!.(mcmc_states, outputs)

    (mcmc_states = mcmc_states, outputs = outputs)
end
