mutable struct OnlyBurninTuner{
    S<:MCMCBasicStats
} <: AbstractMCMCTuner
    stats::S
end

function OnlyBurninTuner(
    chain::MCMCIterator
)
    OnlyBurninTuner(MCMCBasicStats(chain))
end


struct OnlyBurninTunerConfig <: BAT.AbstractMCMCTuningStrategy end

(config::OnlyBurninTunerConfig)(chain::MCMCIterator; kwargs...) = OnlyBurninTuner(chain)

isviable(tuner::OnlyBurninTuner, chain::MCMCIterator) = true

function tuning_init!(tuner::OnlyBurninTuner, chain::MCMCIterator)
    nothing
end


function run_tuning_iterations!(
    callbacks,
    tuners::AbstractVector{<:OnlyBurninTuner},
    chains::AbstractVector{<:MCMCIterator};
    max_nsamples::Int64 = Int64(1000),
    max_nsteps::Int64 = Int64(10000),
    max_time::Float64 = Inf
)
    user_callbacks = mcmc_callback_vector(callbacks, eachindex(chains))

    combined_callbacks = broadcast(tuners, user_callbacks) do tuner, user_callback
        (level, chain) -> begin
            if level == 1
                get_samples!(tuner.stats, chain, true)
            end
            user_callback(level, chain)
        end
    end

    mcmc_iterate!(combined_callbacks, chains, max_nsamples = max_nsamples, max_nsteps = max_nsteps, max_time = max_time)
    nothing
end


# run tuning iterations without any tuning update
function run_tuning_cycle!(
    callbacks,
    tuners::AbstractVector{<:OnlyBurninTuner},
    chains::AbstractVector{<:MCMCIterator};
    kwargs...
)
    run_tuning_iterations!(callbacks, tuners, chains; kwargs...)
    nothing
end


# just burnin, no tuning
function mcmc_tune_burnin!(
    callbacks,
    tuners::AbstractVector{<:OnlyBurninTuner},
    chains::AbstractVector{<:MCMCIterator},
    convergence_test::MCMCConvergenceTest,
    burnin_strategy::MCMCBurninStrategy;
    strict_mode::Bool = false
)
    @info "Begin burnin of $(length(tuners)) MCMC chain(s)."

    nchains = length(chains)

    user_callbacks = mcmc_callback_vector(callbacks, eachindex(chains))

    cycles = zero(Int)
    successful = false
    while !successful && cycles < burnin_strategy.max_ncycles
        cycles += 1
        run_tuning_cycle!(
            user_callbacks, tuners, chains;
            max_nsamples = burnin_strategy.max_nsamples_per_cycle,
            max_nsteps = burnin_strategy.max_nsteps_per_cycle,
            max_time = burnin_strategy.max_time_per_cycle
        )

        new_stats = [x.stats for x in tuners] # ToDo: Find more generic abstraction
        ct_result = check_convergence!(convergence_test, chains, new_stats)

        nconverged = count(c -> c.info.converged, chains)
        successful = (nconverged == nchains)

        for i in eachindex(user_callbacks, tuners)
            user_callbacks[i](1, tuners[i])
        end

        @info "MCMC burnin cycle $cycles finished, $nchains chains, $nconverged converged."
    end

    if successful
        @info "MCMC burnin of $nchains chains successful after $cycles cycle(s)."
    else
        msg = "MCMC burnin of $nchains chains aborted after $cycles cycle(s)."
        if strict_mode
            @error msg
        else
            @warn msg
        end
    end

    successful
end
