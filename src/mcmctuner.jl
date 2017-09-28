# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT.Logging


abstract type AbstractMCMCTunerConfig end
export AbstractMCMCTunerConfig


abstract type AbstractMCMCTuner end
export AbstractMCMCTuner



function mcmc_auto_tune!(
    callback,
    chains::AbstractVector{<:MCMCChain},
    exec_context::ExecContext = ExecContext(),
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(first(chains).algorithm),
    convergence_test::MCMCConvergenceTest = GRConvergence();
    init_proposal::Bool = true,
    max_nsamples_per_cycle::Int64 = Int64(1000),
    max_nsteps_per_cycle::Int = 10000,
    max_time_per_cycle::Float64 = Inf,
    max_ncycles::Int = 30,
    granularity::Int = 1,
    ll::LogLevel = LOG_INFO
)
    @log_info "Starting tuning of $(length(chains)) chain(s)."

    nchains = length(chains)
    tuners = [tuner_config(c, init_proposal = init_proposal) for c in chains]

    cycle = 0
    successful = false
    while !successful && cycle <= max_ncycles
        for tuner in tuners
            run_tuning_cycle!(
                callback, tuner, exec_context,
                max_nsamples = max_nsamples_per_cycle, max_nsteps = max_nsteps_per_cycle,
                max_time = max_time_per_cycle, granularity = granularity, ll = ll+1
            )
        end

        stats = [x.stats for x in tuners]
        ct_result = check_convergence!(convergence_test, chains, stats, ll = ll+1)

        ntuned = count(c -> c.info.tuned, chains)
        nconverged = count(c -> c.info.converged, chains)
        successful = (ntuned == nconverged == nchains)

        @log_msg ll+1 "MCMC Tuning cycle $cycle finished, $nchains chains, $ntuned tuned, $nconverged converged."
        cycle += 1
    end

    if successful
        @log_msg ll "MCMC tuning of $nchains chains successful after $cycle cycle(s)."
    else
        @log_msg ll-1 "MCMC tuning of $nchains chains aborted after $cycle cycle(s)."
    end

    successful
end

export mcmc_auto_tune!

