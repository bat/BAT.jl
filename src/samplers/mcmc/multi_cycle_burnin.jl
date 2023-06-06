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
    chains::AbstractVector{<:MCMCIterator},
    burnin_alg::MCMCMultiCycleBurnin,
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

        new_outputs = DensitySampleVector.(chains)

        next_cycle!.(chains)

        tuning_reinit!.(tuners, chains, burnin_alg.nsteps_per_cycle)

        mcmc_iterate!(
            new_outputs, chains, tuners,
            max_nsteps = burnin_alg.nsteps_per_cycle,
            nonzero_weights = nonzero_weights,
            callback = callback
        )

        tuning_update_cycle!.(tuners, chains, new_outputs)
        isnothing(outputs) || append!.(outputs, new_outputs)

        check_convergence!(chains, new_outputs, convergence_test)

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

        next_cycle!.(chains)

        mcmc_iterate!(
            outputs, chains,
            max_nsteps = burnin_alg.nsteps_final,
            nonzero_weights = nonzero_weights,
            callback = callback
        )
    end

    successful
end
