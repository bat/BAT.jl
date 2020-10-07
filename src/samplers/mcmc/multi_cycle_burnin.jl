# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    MCMCMultiCycleBurnin <: MCMCBurninAlgorithm

A multi-cycle MCMC burn-in algorithm.

Fields:
* `max_nsamples_per_cycle`: Maximum number of MCMC samples to generate per
  cycle, defaults to `1000`. Definition of a sample depends on MCMC algorithm.
* `max_nsteps_per_cycle`: Maximum number of MCMC steps per cycle, defaults
  to `10000`. Definition of a step depends on MCMC algorithm.
* `max_time_per_cycle`: Maximum wall-clock time to spend per cycle, in
  seconds. Defaults to `Inf`.
* `max_ncycles`: Maximum number of cycles.
"""
@with_kw struct MCMCMultiCycleBurnin <: MCMCBurninAlgorithm
    max_nsamples_per_cycle::Int64 = 1000
    max_nsteps_per_cycle::Int64 = 10000
    max_time_per_cycle::Float64 = Inf
    max_ncycles::Int = 30
end

export MCMCMultiCycleBurnin


function mcmc_burnin!(
    outputs::Union{AbstractVector{<:DensitySampleVector},Nothing},
    tuners::AbstractVector{<:AbstractMCMCTunerInstance},
    chains::AbstractVector{<:MCMCIterator},
    burnin_alg::MCMCMultiCycleBurnin,
    convergence_test::MCMCConvergenceTest,
    strict_mode::Bool,
    nonzero_weights::Bool,
    callback::Function
)
    @info "Begin tuning of $(length(tuners)) MCMC chain(s)."

    nchains = length(chains)

    cycles = zero(Int)
    successful = false
    while !successful && cycles < burnin_alg.max_ncycles
        cycles += 1

        new_outputs = DensitySampleVector.(chains)

        mcmc_iterate!(
            new_outputs,
            chains,
            max_nsamples = burnin_alg.max_nsamples_per_cycle,
            max_nsteps = burnin_alg.max_nsteps_per_cycle,
            max_time = burnin_alg.max_time_per_cycle,
            nonzero_weights = nonzero_weights,
            callback = callback
        )

        tuning_update!.(tuners, chains, new_outputs)
        isnothing(outputs) || append!.(outputs, new_outputs)

        ct_result = check_convergence!(convergence_test, chains, new_outputs)

        ntuned = count(c -> c.info.tuned, chains)
        nconverged = count(c -> c.info.converged, chains)
        successful = (ntuned == nconverged == nchains)

        callback(Val(:mcmc_burnin), tuners, chains)

        @info "MCMC Tuning cycle $cycles finished, $nchains chains, $ntuned tuned, $nconverged converged."

        next_cycle!.(chains)
    end

    if successful
        @info "MCMC tuning of $nchains chains successful after $cycles cycle(s)."
    else
        msg = "MCMC tuning of $nchains chains aborted after $cycles cycle(s)."
        if strict_mode
            @error msg
        else
            @warn msg
        end
    end

    successful
end
