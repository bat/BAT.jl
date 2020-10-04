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
@with_kw struct MCMCMultiCycleBurnin
    max_nsamples_per_cycle::Int64 = 1000
    max_nsteps_per_cycle::Int64 = 10000
    max_time_per_cycle::Float64 = Inf
    max_ncycles::Int = 30
end

export MCMCMultiCycleBurnin


function mcmc_burnin!(
    callbacks,
    tuners::AbstractVector{<:AbstractMCMCTunerInstance},
    chains::AbstractVector{<:MCMCIterator},
    burnin_strategy::MCMCMultiCycleBurnin,
    convergence_test::MCMCConvergenceTest;
    strict_mode::Bool = false
)
    @info "Begin tuning of $(length(tuners)) MCMC chain(s)."

    nchains = length(chains)

    user_callbacks = mcmc_callback_vector(callbacks, eachindex(chains))

    cycles = zero(Int)
    successful = false
    while !successful && cycles < burnin_strategy.max_ncycles
        old_stats = [x.stats for x in tuners] # ToDo: Find more generic abstraction
        stats_reweight_factors = [x.config.r for x in tuners] # ToDo: Find more generic abstraction
        reweight_relative!.(old_stats, stats_reweight_factors)
        # empty!.(old_stats)

        cycles += 1
        run_tuning_cycle!(
            user_callbacks, tuners, chains;
            max_nsamples = burnin_strategy.max_nsamples_per_cycle,
            max_nsteps = burnin_strategy.max_nsteps_per_cycle,
            max_time = burnin_strategy.max_time_per_cycle
        )

        new_stats = [x.stats for x in tuners] # ToDo: Find more generic abstraction
        ct_result = check_convergence!(convergence_test, chains, new_stats)

        ntuned = count(c -> c.info.tuned, chains)
        nconverged = count(c -> c.info.converged, chains)
        successful = (ntuned == nconverged == nchains)

        for i in eachindex(user_callbacks, tuners)
            user_callbacks[i](1, tuners[i])
        end

        @info "MCMC Tuning cycle $cycles finished, $nchains chains, $ntuned tuned, $nconverged converged."
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
