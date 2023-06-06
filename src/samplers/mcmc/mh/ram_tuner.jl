# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Add literature references to RAMTuning docstring.

"""
    struct RAMTuning <: MHProposalDistTuning

RAM tuning algorithm.

Based on [M. Vihola, Robust adaptive Metropolis algorithm with coerced acceptance rate](https://doi.org/10.1007/s11222-011-9269-5).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct RAMTuning <: MHProposalDistTuning
    "Controls the weight given to new covariance information in adapting the
    proposal distribution, value must be between 1/2 and 1."
    γ = 2/3

    "Metropolis-Hastings acceptance ratio target, tuning will try to adapt
    the proposal distribution to bring the acceptance ratio inside this interval."
    α::IntervalSets.ClosedInterval{Float64} = ClosedInterval(0.15, 0.35)
end

export RAMTuning



mutable struct RAMTunerInstance{
    S<:MCMCBasicStats
} <: AbstractMCMCTunerInstance
    config::RAMTuning
    iteration::Int
    scale::Float64
end

(tuning::RAMTuning)(chain::MHIterator) = RAMTunerInstance(tuning, chain)


function RAMTunerInstance(tuning::RAMTuning, chain::MHIterator)
    m = totalndof(getmeasure(chain))
    scale = 2.38^2 / m
    RAMTunerInstance(tuning, MCMCBasicStats(chain), 1, scale)
end

function tuning_init!(tuner::RAMTunerInstance, chain::MHIterator, max_nsteps::Integer)
    Σ_unscaled = _approx_cov(getmeasure(chain))
    Σ = Σ_unscaled * tuner.scale
    
    chain.proposaldist = set_cov(chain.proposaldist, Σ)

    nothing
end


tuning_reinit!(tuner::RAMTunerInstance, chain::MCMCIterator, max_nsteps::Integer) = nothing


function tuning_postinit!(tuner::RAMTunerInstance, chain::MHIterator, samples::DensitySampleVector)
    # The very first samples of a chain can be very valuable to init tuner
    # stats, especially if the chain gets stuck early after:
    stats = tuner.stats
    append!(stats, samples)
end


function tuning_update!(tuner::RAMTunerInstance, chain::MHIterator, samples::DensitySampleVector)
    stats = tuner.stats
    stats_reweight_factor = tuner.config.r
    reweight_relative!(stats, stats_reweight_factor)
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

    chain.proposaldist = set_cov(chain.proposaldist, Σ_new)
    tuner.iteration += 1

    nothing
end

tuning_finalize!(tuner::RAMTunerInstance, chain::MCMCIterator) = nothing

tuning_callback(::RAMTunerInstance) = nop_func
