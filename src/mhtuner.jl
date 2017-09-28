# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT.Logging

struct ProposalCovTunerConfig <: AbstractMCMCTunerConfig
    lambda::Float64 # e.g. 0.5
end

ProposalCovTunerConfig() = ProposalCovTunerConfig(0.5)

export ProposalCovTunerConfig



mutable struct ProposalCovTuner{
    C<:MCMCChain{<:MetropolisHastings},
    S<:MCMCBasicStats
} <: AbstractMCMCTuner
    config::ProposalCovTunerConfig
    chain::C
    stats::S
    iteration::Int
    scale::Float64
end

export ProposalCovTuner

function ProposalCovTuner(
    config::ProposalCovTunerConfig,
    chain::MCMCChain{<:MetropolisHastings},
    init_proposal::Bool = true
)
    m = nparams(chain)
    scale = 2.38^2 / m
    tuner = ProposalCovTuner(config, chain, MCMCBasicStats(chain), 1, scale)

    if init_proposal
        tuning_init_proposal!(tuner)
    end

    tuner
end


AbstractMCMCTunerConfig(algorithm::MetropolisHastings) = ProposalCovTunerConfig()

AbstractMCMCTuner(config::ProposalCovTunerConfig, chain::MCMCChain{<:MetropolisHastings}, init_proposal::Bool = true) =
    ProposalCovTuner(config, chain, init_proposal)


function tuning_init_proposal!(tuner::ProposalCovTuner)
    chain = tuner.chain

    # ToDo: Generalize for non-hypercube bounds
    bounds = chain.target.bounds
    flat_var = (bounds.vol.hi - bounds.vol.lo).^2 / 12

    m = length(flat_var)
    Σ_unscaled = full(PDiagMat(flat_var))
    Σ = Σ_unscaled * tuner.scale

    next_cycle!(chain)
    chain.state.pdist = set_cov!(chain.state.pdist, Σ)

    chain
end


function tuning_update!(tuner::ProposalCovTuner; ll::LogLevel = LOG_NONE)
    chain = tuner.chain
    stats = tuner.stats

    α_min = 0.15
    α_max = 0.35

    c_min = 1e-4
    c_max = 1e2

    β = 1.5

    state = chain.state

    t = tuner.iteration
    λ = tuner.config.lambda
    c = tuner.scale
    Σ_old = full(get_cov(state.pdist))

    S = convert(Array, stats.param_stats.cov)
    a_t = 1 / t^λ
    new_Σ_unscal = (1 - a_t) * (Σ_old/c) + a_t * S

    α = acceptance_ratio(state)

    if α_min <= α <= α_max
        chain.info = set_tuned(chain.info, true)
        @log_msg ll "MCMC chain $(chain.info.id) tuned, acceptance ratio = $α"
    else
        chain.info = set_tuned(chain.info, false)
        @log_msg ll "MCMC chain $(chain.info.id) *not* tuned, acceptance ratio = $α"

        if α > α_max && c < c_max
            tuner.scale = c * β
        elseif α < α_min && c > c_min
            tuner.scale = c / β
        end
    end

    Σ_new = full(Hermitian(new_Σ_unscal * tuner.scale))

    next_cycle!(chain)
    state.pdist = set_cov!(state.pdist, Σ_new)
    tuner.iteration += 1

    chain
end


function run_tuning_cycle!(
    callback,
    tuner::ProposalCovTuner,
    exec_context::ExecContext = ExecContext();
    max_nsamples::Int64 = Int64(1000),
    max_nsteps::Int = 10000,
    max_time::Float64 = Inf,
    granularity::Int = 1,
    ll::LogLevel = LOG_NONE
)

    mcmc_iterate!(tuner.chain, exec_context, max_nsamples = max_nsamples, max_nsteps = max_nsteps, max_time = max_time, granularity = granularity) do chain
        push!(tuner.stats, chain)
        callback(tuner)
    end
    tuning_update!(tuner; ll = ll)
end





function mcmc_auto_tune!(
    callback,
    chains::AbstractVector{<:MCMCChain},
    exec_context::ExecContext = ExecContext(),
    tuner_config::AbstractMCMCTunerConfig = AbstractMCMCTunerConfig(first(chains).algorithm),
    convergence_test::MCMCConvergenceTest = GRConvergence();
    max_nsamples_per_cycle::Int64 = Int64(1000),
    max_nsteps_per_cycle::Int = 10000,
    max_time_per_cycle::Float64 = Inf,
    max_ncycles::Int = 30,
    granularity::Int = 1,
    ll::LogLevel = LOG_INFO
)
    @log_info "Starting tuning of $(length(chains)) chain(s)."

    nchains = length(chains)

    cycle = 0
    successful = false
    while !successful && cycle <= max_ncycles
        run_tuning_cycle!(
            callback, tuner, exec_context,
            max_nsamples = max_nsamples_per_cycle, max_nsteps = max_nsteps_per_cycle,
            max_time = max_time_per_cycle, granularity = granularity, ll = ll
        )

        stats = [x.stats for x in tuners]
        ct_result = check_convergence!(convergence_test, chains, stats, ll = ll)

        ntuned = count(c -> c.info.tuned, chains)
        nconverged = count(c -> c.info.converged, chains)
        successful = (ntuned == nconverged == nchains)

        @log_msg ll+1 "MCMC Tuning cycle $cycle finished, $nchains chains, $ntuned tuned, $nconverged converged."
    end

    if successful
        @log_msg ll "MCMC tuning of $nchains chains successful after $cycle cycle(s)."
    else
        @log_msg ll-1 "MCMC tuning of $nchains chains aborted after $cycle cycle(s)."
    end

    successful
end

