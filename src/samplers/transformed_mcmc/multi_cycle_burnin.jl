# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct TransformedMCMCMultiCycleBurnin <: TransformedMCMCBurninAlgorithm

A multi-cycle MCMC burn-in algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct TransformedMCMCMultiCycleBurnin <: TransformedMCMCBurninAlgorithm
    nsteps_per_cycle::Int64 = 10000
    max_ncycles::Int = 30
    nsteps_final::Int64 = div(nsteps_per_cycle, 10)
end

export TransformedMCMCMultiCycleBurnin


function mcmc_burnin!(
    outputs::Union{DensitySampleVector,Nothing},
    chains::AbstractVector{<:MCMCIterator},
    tuners::AbstractVector{<:TransformedAbstractMCMCTunerInstance},
    temperers::AbstractVector{<:TransformedMCMCTemperingInstance},
    burnin_alg::TransformedMCMCMultiCycleBurnin,
    convergence_test::ConvergenceTest,
    strict_mode::Bool,
    nonzero_weights::Bool,
    callback::Function
)
    nchains = length(chains)

    @info "Begin tuning of $nchains MCMC chain(s)."

    cycles = zero(Int)
    successful = false
    while !successful && cycles < burnin_alg.max_ncycles
        cycles += 1

        next_cycle!.(chains)

        tuning_reinit!.(tuners, chains, burnin_alg.nsteps_per_cycle)

        desc_string = string("Burnin cycle ", cycles, "/max_cycles=", burnin_alg.max_ncycles," for nchains=", length(chains))
        progress_meter = ProgressMeter.Progress(length(chains)*burnin_alg.nsteps_per_cycle, desc=desc_string, barlen=80-length(desc_string), dt=0.1)
    
        transformed_mcmc_iterate!(
            chains, tuners, temperers,
            max_nsteps = burnin_alg.nsteps_per_cycle,
            nonzero_weights = nonzero_weights,
            callback = (kwargs...) -> let pm=progress_meter; ProgressMeter.next!(progress_meter) ; end,
        )
        ProgressMeter.finish!(progress_meter)

        new_outputs = getproperty.(chains, :samples)

        tuning_update!.(tuners, chains, new_outputs)
        
        isnothing(outputs) || append!(outputs, reduce(vcat, new_outputs))

        transformed_check_convergence!(chains, new_outputs, convergence_test) # TODO AC: Rename

        # check_tuned/update_tuners...
        ntuned = count(c -> c.info.tuned, chains)
        nconverged = count(c -> c.info.converged, chains)
        successful = (ntuned == nconverged == nchains)

        callback(Val(:mcmc_burnin), tuners, chains)

        @info "MCMC Tuning cycle $cycles finished, $nchains chains, $ntuned tuned, $nconverged converged."
    end

    tuning_finalize!.(tuners, chains)
    
    if successful
        @info "MCMC tuning of $nchains chains successful after $cycles cycle(s)."
    else
        msg = "MCMC tuning of $nchains chains aborted after $cycles cycle(s)."
        if strict_mode
            throw(ErrorException(msg))
        else
            @warn msg
        end
    end

    if burnin_alg.nsteps_final > 0
        @info "Running post-tuning stabilization steps for $nchains MCMC chain(s)."

        # turn off tuning
        next_cycle!.(chains)
        tuners = TransformedMCMCNoOpTuning().(chains)

        # TODO AC: what about tempering?
        
        transformed_mcmc_iterate!(
            chains, tuners, temperers,
            max_nsteps = burnin_alg.nsteps_final,
            nonzero_weights = nonzero_weights,
            callback = callback
        )
    end

    successful
end
