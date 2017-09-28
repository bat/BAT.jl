# This file is a part of BAT.jl, licensed under the MIT License (MIT).


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


function tuning_update!(tuner::ProposalCovTuner)
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

    new_c = if α > α_max && c < c_max
        convert(typeof(c), c * β)
    elseif α < α_min && c > c_min
        convert(typeof(c), c / β)
    else
        c
    end

    tuner.scale = new_c

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
    granularity::Int = 1
)

    mcmc_iterate!(tuner.chain, exec_context, max_nsamples = max_nsamples, max_nsteps = max_nsteps, max_time = max_time, granularity = granularity) do chain
        push!(tuner.stats, chain)
        callback(tuner)
    end
    tuning_update!(tuner)
end



# function mcmc_auto_tune!(
#     callback,
#     chains::MCMCChain{<:MetropolisHastings},
#     exec_context::ExecContext = ExecContext(),
#     chains_stats::AbstractVector{<:MCMCBasicStats};
#     max_nsamples_per_cycle::Int64 = Int64(1),
#     max_nsteps_per_cycle::Int = 10000,
#     max_nsamples_per_cycle::Int64 = 1000,
#     max_ncycles::Int = 30,
#     max_time::Float64 = Inf,
#     granularity::Int = 1
#
#     MCMCConvergenceTest
# )
