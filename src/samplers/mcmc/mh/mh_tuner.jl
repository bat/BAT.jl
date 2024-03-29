# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Add literature references to AdaptiveMHTuning docstring.

"""
    struct AdaptiveMHTuning <: MHProposalDistTuning

Adaptive MCMC tuning strategy for Metropolis-Hastings samplers.

Adapts the proposal function based on the acceptance ratio and covariance
of the previous samples.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct AdaptiveMHTuning <: MHProposalDistTuning
    "Controls the weight given to new covariance information in adapting the
    proposal distribution."
    λ::Float64 = 0.5

    "Metropolis-Hastings acceptance ratio target, tuning will try to adapt
    the proposal distribution to bring the acceptance ratio inside this interval."
    α::IntervalSets.ClosedInterval{Float64} = ClosedInterval(0.15, 0.35)

    "Controls how much the spread of the proposal distribution is
    widened/narrowed depending on the current MH acceptance ratio."
    β::Float64 = 1.5

    "Interval for allowed scale/spread of the proposal distribution."
    c::IntervalSets.ClosedInterval{Float64} = ClosedInterval(1e-4, 1e2)

    "Reweighting factor. Take accumulated sample statistics of previous
    tuning cycles into account with a relative weight of `r`. Set to
    `0` to completely reset sample statistics between each tuning cycle."
    r::Real = 0.5
end

export AdaptiveMHTuning



mutable struct ProposalCovTuner{
    S<:MCMCBasicStats
} <: AbstractMCMCTunerInstance
    config::AdaptiveMHTuning
    stats::S
    iteration::Int
    scale::Float64
end

(tuning::AdaptiveMHTuning)(chain::MHIterator) = ProposalCovTuner(tuning, chain)


function ProposalCovTuner(tuning::AdaptiveMHTuning, chain::MHIterator)
    m = totalndof(varshape(mcmc_target(chain)))
    scale = 2.38^2 / m
    ProposalCovTuner(tuning, MCMCBasicStats(chain), 1, scale)
end


function _cov_with_fallback(m::BATMeasure)
    global g_state = m
    @assert false
    rng = _bat_determ_rng()
    T = float(eltype(rand(rng, m)))
    n = totalndof(varshape(m))
    C = fill(T(NaN), n, n)
    try
        C[:] = cov(m)
    catch err
        if err isa MethodError
            C[:] = cov(nestedview(rand(rng, m, 10^5)))
        else
            throw(err)
        end
    end
    return C
end


function tuning_init!(tuner::ProposalCovTuner, chain::MHIterator, max_nsteps::Integer)
    Σ_unscaled = get_cov(chain.proposaldist)
    Σ = Σ_unscaled * tuner.scale
    
    chain.proposaldist = set_cov(chain.proposaldist, Σ)

    nothing
end


tuning_reinit!(tuner::ProposalCovTuner, chain::MCMCIterator, max_nsteps::Integer) = nothing


function tuning_postinit!(tuner::ProposalCovTuner, chain::MHIterator, samples::DensitySampleVector)
    # The very first samples of a chain can be very valuable to init tuner
    # stats, especially if the chain gets stuck early after:
    stats = tuner.stats
    append!(stats, samples)
end


function tuning_update!(tuner::ProposalCovTuner, chain::MHIterator, samples::DensitySampleVector)
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

tuning_finalize!(tuner::ProposalCovTuner, chain::MCMCIterator) = nothing

tuning_callback(::ProposalCovTuner) = nop_func
