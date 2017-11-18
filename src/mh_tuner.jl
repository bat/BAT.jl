# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@with_kw struct ProposalCovTunerConfig <: AbstractMCMCTunerConfig
    λ::Float64 = 0.5
    α::IntervalSets.ClosedInterval{Float64} = ClosedInterval(0.15, 0.35)
    β::Float64 = 1.5
    c::IntervalSets.ClosedInterval{Float64} = ClosedInterval(1e-4, 1e2)
end

export ProposalCovTunerConfig


AbstractMCMCTunerConfig(algorithm::MetropolisHastings) = ProposalCovTunerConfig()

(config::ProposalCovTunerConfig)(chain::MCMCIterator{<:MetropolisHastings}; init_proposal::Bool = true) =
    ProposalCovTuner(config, chain, init_proposal)



mutable struct ProposalCovTuner{
    C<:MCMCIterator{<:MetropolisHastings},
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
    chain::MCMCIterator{<:MetropolisHastings},
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


isviable(tuner::ProposalCovTuner) = nsamples(tuner.chain.state) >= 2


function tuning_init_proposal!(tuner::ProposalCovTuner)
    chain = tuner.chain

    # ToDo: Generalize, currently limited to HyperRectBounds
    bounds = param_bounds(chain.target)
    vol = spatialvolume(bounds)
    flat_var = (vol.hi - vol.lo).^2 / 12

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
    config = tuner.config

    α_min = minimum(config.α)
    α_max = maximum(config.α)

    c_min = minimum(config.c)
    c_max = maximum(config.c)

    β = config.β

    state = chain.state

    t = tuner.iteration
    λ = config.λ
    c = tuner.scale
    Σ_old = full(get_cov(state.pdist))

    S = convert(Array, stats.param_stats.cov)
    a_t = 1 / t^λ
    new_Σ_unscal = (1 - a_t) * (Σ_old/c) + a_t * S

    α = acceptance_ratio(state)

    if α_min <= α <= α_max
        chain.tuned = true
        @log_msg ll "MCMC chain $(chain.id) tuned, acceptance ratio = $α"
    else
        chain.tuned = false
        @log_msg ll "MCMC chain $(chain.id) *not* tuned, acceptance ratio = $α"

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
    callbacks,
    tuners::AbstractVector{<:ProposalCovTuner},
    exec_context::ExecContext = ExecContext();
    ll::LogLevel = LOG_NONE,
    kwargs...
)
    run_tuning_iterations!(callbacks, tuners, exec_context; ll=ll, kwargs...)
    tuning_update!.(tuners; ll = ll)
    nothing
end


function run_tuning_iterations!(
    callbacks,
    tuners::AbstractVector{<:ProposalCovTuner},
    exec_context::ExecContext;
    max_nsamples::Int64 = Int64(1000),
    max_nsteps::Int64 = Int64(10000),
    max_time::Float64 = Inf,
    ll::LogLevel = LOG_NONE
)
    chains = map(x -> x.chain, tuners)
    user_callbacks = mcmc_callback_vector(callbacks, chains)

    combined_callbacks = broadcast(tuners, user_callbacks) do tuner, user_callback
        (level, chain) -> begin
            if level == 1
                push!(tuner.stats, chain)
            end
            user_callback(level, chain)
        end
    end

    mcmc_iterate!(combined_callbacks, chains, exec_context, max_nsamples = max_nsamples, max_nsteps = max_nsteps, max_time = max_time)
    nothing
end
