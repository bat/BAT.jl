# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Revise has trouble revising if @kwargs type definitions are in here directly:
include("mh_tuner_types.jl")


mutable struct ProposalCovTuner{
    S<:MCMCBasicStats
} <: AbstractMCMCTunerInstance
    config::AdaptiveMHTuning
    stats::S
    iteration::Int
    scale::Float64
end


function ProposalCovTuner(
    config::AdaptiveMHTuning,
    chain::MHIterator
)
    m = totalndof(getdensity(chain))
    scale = 2.38^2 / m
    ProposalCovTuner(config, MCMCBasicStats(chain), 1, scale)
end



_approx_cov(target::Distribution) = cov(target)
_approx_cov(target::DistLikeDensity) = cov(target)
_approx_cov(target::AbstractPosteriorDensity) = cov(getprior(target))

function tuning_init!(tuner::ProposalCovTuner, chain::MHIterator)
    Σ_unscaled = _approx_cov(getdensity(chain))
    Σ = Σ_unscaled * tuner.scale

    next_cycle!(chain)
    chain.proposaldist = set_cov(chain.proposaldist, Σ)

    nothing
end


function tuning_update!(tuner::ProposalCovTuner, chain::MHIterator, samples::DensitySampleVector)
    stats = tuner.stats
    stats_reweight_factor = tuner.config.r
    reweight_relative!.(stats, stats_reweight_factor)
    # empty!.(stats)
    append!(stats, samples)

    config = tuner.config

    α_min = minimum(config.α)
    α_max = maximum(config.α)

    c_min = minimum(config.c)
    c_max = maximum(config.c)

    β = config.β

    t = tuner.iteration
    λ = config.λ
    c = tuner.scale
    Σ_old = Matrix(get_cov(chain.proposaldist))

    S = convert(Array, stats.param_stats.cov)
    a_t = 1 / t^λ
    new_Σ_unscal = (1 - a_t) * (Σ_old/c) + a_t * S

    α = eff_acceptance_ratio(chain)

    max_log_posterior = stats.logtf_stats.maximum

    if α_min <= α <= α_max
        chain.info = MCMCIteratorInfo(chain.info, tuned = true)
        @debug "MCMC chain $(chain.info.id) tuned, acceptance ratio = $(Float32(α)), proposal scale = $(Float32(c)), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain.info = MCMCIteratorInfo(chain.info, tuned = false)
        @debug "MCMC chain $(chain.info.id) *not* tuned, acceptance ratio = $(Float32(α)), proposal scale = $(Float32(c)), max. log posterior = $(Float32(max_log_posterior))"

        if α > α_max && c < c_max
            tuner.scale = c * β
        elseif α < α_min && c > c_min
            tuner.scale = c / β
        end
    end

    Σ_new = new_Σ_unscal * tuner.scale

    next_cycle!(chain)
    chain.proposaldist = set_cov(chain.proposaldist, Σ_new)
    tuner.iteration += 1

    nothing
end
