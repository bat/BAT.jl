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
    outputs::Union{AbstractVector{<:DensitySampleVector},Nothing},
    tuners::AbstractVector{<:AbstractMCMCTunerInstance},
    mc_states::AbstractVector{<:MCMCState},
    sampling::MCMCSampling,
    callback::Function
)
    nchains = length(mc_states)

    @unpack burnin, convergence, strict, nonzero_weights = sampling

    @info "Begin tuning of $nchains MCMC chain(s)."

    cycles = zero(Int)
    successful = false
    while !successful && cycles < burnin.max_ncycles
        cycles += 1

        new_outputs = DensitySampleVector.(mc_states)

        next_cycle!.(mc_states)

        tuning_reinit!.(tuners, mc_states, burnin.nsteps_per_cycle)

        mcmc_iterate!(
            new_outputs, mc_states;
            tuners = tuners,
            max_nsteps = burnin.nsteps_per_cycle,
            nonzero_weights = nonzero_weights
        )

        tuning_update!.(tuners, mc_states, new_outputs)

        isnothing(outputs) || append!.(outputs, new_outputs)

        # ToDo: Convergence tests are a special case, they're not supposed
        # to change any state, so we don't want to use the context of the
        # first chain here. But just making a new context is also not ideal.
        # Better copy the context of the first chain and replace the RNG
        # with a new one in the future:
        check_convergence!(mc_states, new_outputs, convergence, BATContext())

        ntuned = count(c -> c.info.tuned, mc_states)
        nconverged = count(c -> c.info.converged, mc_states)
        successful = (ntuned == nconverged == nchains)

        callback(Val(:mcmc_burnin), tuners, mc_states)

        @info "MCMC Tuning cycle $cycles finished, $nchains chains, $ntuned tuned, $nconverged converged."
    end

    tuning_finalize!.(tuners, mc_states)
    
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

        next_cycle!.(mc_states)

        mcmc_iterate!(
            outputs, mc_states,
            max_nsteps = burnin.nsteps_final,
            nonzero_weights = nonzero_weights
        )
    end

    #TODO: MD, Discuss: Where/When Tempering? 

    successful
end
