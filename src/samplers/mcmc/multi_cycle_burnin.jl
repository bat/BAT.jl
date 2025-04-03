# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct MCMCMultiCycleBurnin <: MCMCBurninAlgorithm

A multi-cycle MCMC burn-in algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MCMCMultiCycleBurnin <: MCMCBurninAlgorithm
    nsteps_per_cycle::Int64 = 10000
    max_ncycles::Int = 30
    nsteps_final::Int64 = div(nsteps_per_cycle, 10)
end

export MCMCMultiCycleBurnin


function mcmc_burnin!(
    outputs::Union{AbstractVector{AbstractVector{<:DensitySampleVector}}, Nothing},
    mcmc_states::AbstractVector{<:MCMCState},
    samplingalg::TransformedMCMC,
    callback::Function
)
    nchains = length(mcmc_states)

    @unpack burnin, convergence, strict, nonzero_weights = samplingalg

    @info "Begin tuning of $nchains MCMC chain(s)."

    cycles = zero(Int)
    successful = false

    while !successful && cycles < burnin.max_ncycles
        cycles += 1

        new_outputs = _empty_chain_outputs.(mcmc_states)

        next_cycle!.(mcmc_states)

        mcmc_tuning_reinit!!.(mcmc_states, burnin.nsteps_per_cycle)

        mcmc_states = mcmc_iterate!!(
            new_outputs, mcmc_states;
            max_nsteps = burnin.nsteps_per_cycle,
            nonzero_weights = nonzero_weights
        )
        
        mcmc_states = mcmc_tune_post_cycle!!.(mcmc_states, new_outputs)

        isnothing(outputs) || append!.(outputs, new_outputs)

        # TODO, MD: Rewrite this via append!
        merged_outputs = [merge(walker_output...) for walker_output in new_outputs]
        
        # ToDo: Convergence tests are a special case, they're not supposed
        # to change any state, so we don't want to use the context of the
        # first chain here. But just making a new context is also not ideal.
        # Better copy the context of the first chain and replace the RNG
        # with a new one in the future:
        check_convergence!(mcmc_states, merged_outputs, convergence, BATContext())

        ntuned = count(mcmc_state -> mcmc_state.chain_state.info.tuned, mcmc_states)
        nconverged = count(mcmc_state -> mcmc_state.chain_state.info.converged, mcmc_states)
        successful = (ntuned == nconverged == nchains)

        callback(Val(:mcmc_burnin), mcmc_states)

        @info "MCMC Tuning cycle $cycles finished, $nchains chains, $ntuned tuned, $nconverged converged."
    end

    mcmc_tuning_finalize!!.(mcmc_states)
    
    if successful
        @info "MCMC tuning of $nchains chains successful after $cycles cycle(s)."
    else
        msg = "MCMC tuning of $nchains chains aborted after $cycles cycle(s)."
        if strict
            throw(ErrorException(msg))
        else
            @warn msg
        end
    end

    if burnin.nsteps_final > 0
        @info "Running post-tuning stabilization steps for $nchains MCMC chain(s)."

        next_cycle!.(mcmc_states)

        mcmc_states = mcmc_iterate!!(
            outputs, mcmc_states;
            max_nsteps = burnin.nsteps_final,
            nonzero_weights = nonzero_weights
        )
    end

    #TODO: MD, Discuss: Where/When Tempering? 

    return mcmc_states
end
