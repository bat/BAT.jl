# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT.Logging

struct ProposalCovTunerConfig <: AbstractMCMCTunerConfig
    lambda::Float64 # e.g. 0.5
end

ProposalCovTunerConfig() = ProposalCovTunerConfig(0.5)

export ProposalCovTunerConfig


AbstractMCMCTunerConfig(algorithm::MetropolisHastings) = ProposalCovTunerConfig()

(config::ProposalCovTunerConfig)(chain::MCMCChain{<:MetropolisHastings}; init_proposal::Bool = true) =
    ProposalCovTuner(config, chain, init_proposal)



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
        set_tuned!(chain, true)
        @log_msg ll "MCMC chain $(chain.info.id) tuned, acceptance ratio = $α"
    else
        set_tuned!(chain, false)
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
    ll::LogLevel = LOG_NONE
)
    cbfunc = mcmc_callback(callback)

    mcmc_iterate!(tuner.chain, exec_context, max_nsamples = max_nsamples, max_nsteps = max_nsteps, max_time = max_time) do level, chain
        push!(tuner.stats, chain)
        cbfunc(level, tuner)
    end
    tuning_update!(tuner; ll = ll)
end
