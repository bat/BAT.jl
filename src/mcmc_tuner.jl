# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT.Logging


abstract type AbstractMCMCTunerConfig end
export AbstractMCMCTunerConfig


abstract type AbstractMCMCTuner end
export AbstractMCMCTuner


function mcmc_tune_burnin!(
    callbacks,
    chains::AbstractVector{<:MCMCIterator},
    exec_context::ExecContext = ExecContext(),
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(first(chains).algorithm),
    convergence_test::MCMCConvergenceTest = GRConvergence();
    init_proposal::Bool = true,
    max_nsamples_per_cycle::Int64 = Int64(1000),
    max_nsteps_per_cycle::Int = 10000,
    max_time_per_cycle::Float64 = Inf,
    max_ncycles::Int = 30,
    ll::LogLevel = LOG_INFO
)
    @log_msg ll "Starting tuning of $(length(chains)) MCMC chain(s)."

    user_callbacks = mcmc_callback_vector(callbacks, chains)

    nchains = length(chains)
    tuners = [tuner_config(c, init_proposal = init_proposal) for c in chains]

    cycles = 0
    successful = false
    while !successful && cycles < max_ncycles
        cycles += 1
        run_tuning_cycle!(
            user_callbacks, tuners, exec_context,
            max_nsamples = max_nsamples_per_cycle, max_nsteps = max_nsteps_per_cycle,
            max_time = max_time_per_cycle, ll = ll+2
        )

        stats = [x.stats for x in tuners] # ToDo: Find more generic abstraction
        ct_result = check_convergence!(convergence_test, chains, stats, ll = ll+2)

        ntuned = count(c -> c.tuned, chains)
        nconverged = count(c -> c.converged, chains)
        successful = (ntuned == nconverged == nchains)

        for i in eachindex(user_callbacks, tuners)
            user_callbacks[i](1, tuners[i])
        end

        @log_msg ll+1 "MCMC Tuning cycle $cycles finished, $nchains chains, $ntuned tuned, $nconverged converged."
    end

    if successful
        @log_msg ll "MCMC tuning of $nchains chains successful after $cycles cycle(s)."
    else
        @log_msg LOG_WARNING "MCMC tuning of $nchains chains aborted after $cycles cycle(s)."
    end

    successful
end

export mcmc_tune_burnin!



struct NoOpTunerConfig <: BAT.AbstractMCMCTunerConfig end
export NoOpTunerConfig

(config::NoOpTunerConfig)(chain::MCMCIterator; kwargs...) =
    NoOpTuner(chain)



struct NoOpTuner{C<:MCMCIterator} <: AbstractMCMCTuner
    chain::C
end

export NoOpTuner

function run_tuning_cycle!(
    callbacks,
    tuners::AbstractVector{<:NoOpTuner},
    exec_context::ExecContext = ExecContext();
    ll::LogLevel = LOG_NONE,
    kwargs...
)
    @log_msg ll "NoOpTuner tuning cycle, leaving MCMC chain unchanged."
    nothing
end



function mcmc_tune_burnin!(
    callbacks,
    chains::AbstractVector{<:MCMCIterator},
    exec_context::ExecContext,
    tuner_config::NoOpTunerConfig,
    convergence_test::MCMCConvergenceTest = GRConvergence();
    init_proposal::Bool = true,
    max_nsamples_per_cycle::Int64 = Int64(1000),
    max_nsteps_per_cycle::Int = 10000,
    max_time_per_cycle::Float64 = Inf,
    max_ncycles::Int = 30,
    ll::LogLevel = LOG_INFO
)
    @log_msg ll "Tune/Burn-In with NoOpTuner doing nothing."
end
