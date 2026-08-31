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
    max_init_tries::Int64 = 20
    nsteps_init::Int64 = 250
    initval_alg::InitvalAlgorithm = InitFromTarget()
    strict::Bool = true

    function MCMCRetryInit(max_init_tries, nsteps_init, initval_alg, strict)
        @argcheck max_init_tries > 0
        new(max_init_tries, nsteps_init, initval_alg, strict)
    end
end

export MCMCRetryInit

# TODO: MD, could be generalized for different initalgs?
function apply_trafo_to_init(f_transform::Function, initalg::MCMCRetryInit)
    MCMCRetryInit(
    initalg.max_init_tries,
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

    @info "MCMCRetryInit: trying to generate $nchains viable MCMC chain state(s)."

    initval_alg = init_alg.initval_alg

    rngpart = RNGPartition(get_rng(context), Base.OneTo(nchains))

    mcmc_states = _gen_mcmc_states(samplingalg, target, rngpart, 1:nchains, initval_alg, context)
    outputs = _empty_chain_outputs.(mcmc_states)

    next_cycle!.(mcmc_states)
    mcmc_tuning_init!!.(mcmc_states, init_alg.nsteps_init)
    mcmc_states = mcmc_update_z_position!!.(mcmc_states)

    cycle::Int32 = 1

    success = false

    while !success && cycle <= init_alg.max_init_tries

        @debug "Iterating on $nchains candidate MCMC chain state(s)."

        mcmc_states = mcmc_iterate!!(
            outputs, mcmc_states;
            max_nsteps = clamp(div(init_alg.nsteps_init, 5), 10, 50),
            nonzero_weights = nonzero_weights
        )

        n_unviable_chains = 0
        for i in 1:nchains
            cs = mcmc_states[i].chain_state
            unviable_walkers = findall(
                .!isfinite.(cs.current.x.logd) .|
                (isempty.(outputs[i]) .&& (sum.(getproperty.(outputs[i], :weight)) .< 1))
            )

            if !isempty(unviable_walkers)
                n_unviable_chains += 1
                cycle >= init_alg.max_init_tries && continue

                @debug "Rerolling starting positions for $(length(unviable_walkers)) walkers in chain $i."
                retry_rngpart = RNGPartition(
                    AbstractRNG(rngpart, i), Base.OneTo(init_alg.max_init_tries)
                )
                walker_rngpart = RNGPartition(
                    AbstractRNG(retry_rngpart, cycle), Base.OneTo(samplingalg.nwalkers)
                )
                new_v_init = map(unviable_walkers) do walker
                    new_context = set_rng(context, AbstractRNG(walker_rngpart, walker))
                    bat_initval(target, initval_alg, new_context).result
                end
                new_logd = BAT.checked_logdensityof.(mcmc_target(cs), new_v_init)
                foreach(empty!, view(outputs[i], unviable_walkers))
                cs.current.x.v[unviable_walkers] .= new_v_init
                cs.current.x.logd[unviable_walkers] .= new_logd
                cs.current.x.weight[unviable_walkers] .= 0
                # Bring the z-side representation back in sync with the
                # rerolled x positions:
                mcmc_states[i] = mcmc_update_z_position!!(mcmc_states[i])
                _invalidate_mala_cache!!(mcmc_states[i].chain_state.proposal)
            end
        end

        success = n_unviable_chains == 0
        cycle += 1
    end

    init_alg.strict && !success && error("Failed to generate $nchains viable MCMC chain states")

    @info "Selected $nchains MCMC chain state(s)."

    mcmc_tuning_postinit!!.(mcmc_states, outputs)

    (mcmc_states = mcmc_states, outputs = outputs)
end
