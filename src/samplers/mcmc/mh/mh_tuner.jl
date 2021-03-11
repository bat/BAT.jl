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


mutable struct AdaptiveMHTunerInstance{
    S<:MCMCBasicStats
} <: AbstractMCMCTunerInstance
    config::AdaptiveMHTuning
    stats::S
    iteration::Int
    scale::Float64
end

(tuning::AdaptiveMHTuning)(chain::MHIterator) = AdaptiveMHTunerInstance(tuning, chain)


function AdaptiveMHTunerInstance(tuning::AdaptiveMHTuning, chain::MHIterator)
    m = totalndof(getdensity(chain))

    #Gelman et al. "Efficient Metropolis jumping rules", Bayesian Statistics 5, 599-607, 1996; Section 3.1:
    scale = 2.38^2 / m

    AdaptiveMHTunerInstance(tuning, MCMCBasicStats(chain), 1, scale)
end



_approx_cov(target::Distribution) = cov(unshaped(target))
_approx_cov(target::DistLikeDensity) = cov(target)
_approx_cov(target::AbstractPosteriorDensity) = cov(getprior(target))
_approx_cov(target::BAT.TransformedDensity{<:Any,<:BAT.DistributionTransform}) =
    BAT._approx_cov(target.trafo.target_dist)

function tuning_init!(tuner::AdaptiveMHTunerInstance, chain::MHIterator)
    Σ_unscaled = _approx_cov(getdensity(chain))
    Σ = Σ_unscaled * tuner.scale

    next_cycle!(chain) # ToDo: This would be better placed in the burn-in algorithm
    chain.proposaldist = set_cov(chain.proposaldist, Σ)

    nothing
end


function tuning_postinit!(tuner::AdaptiveMHTunerInstance, chain::MHIterator, samples::DensitySampleVector)
    # The very first samples of a chain can be very valuable to init tuner
    # stats, especially if the chain gets stuck early after:
    stats = tuner.stats
    append!(stats, samples)
end


function tuning_update!(tuner::AdaptiveMHTunerInstance, chain::MHIterator, samples::DensitySampleVector)
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



"""
    struct DynamicMHTuning <: MHProposalDistTuning

*Experimental feature, not part of stable public API.*

Dynmaic MCMC tuning strategy for Metropolis-Hastings samplers.

Adapts the proposal function based on the acceptance ratio and covariance
of the current and previous samples.
        
This is an improved version of [`AdaptiveMHTuning`](@ref) and may become
the new default for MH tuning eventually. Parameters and algorithm are
still subject to change.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct DynamicMHTuning{
    EAL <: EffSampleSizeAlgorithm
} <: MHProposalDistTuning
    "Metropolis-Hastings acceptance ratio target, tuning will try to adapt
    the proposal distribution to bring the acceptance ratio inside this interval."
    α::IntervalSets.ClosedInterval{Float64} = ClosedInterval(0.15, 0.35)

    "Reweighting factor. Take accumulated sample statistics of previous
    tuning cycles into account with a relative weight of `r`. Set to
    `0` to completely reset sample statistics between each tuning cycle."
    r::Real = 0.5

    "Controls evolution of the weight of previous sample statistics,
    effectively reduces `r` over time."
    λ ::Float64 = 0.5

    "Initial proposal scale times number of degrees of freedom."
    s::Float64 = 1

    "Controls how much the spread of the proposal distribution is
    widened/narrowed depending on the current MH acceptance ratio."
    β::Float64 = 4

    "Controls the transition curve that controls the application of β."
    γ::Float64 = 0.5

    "Interval for allowed scale/spread of the proposal distribution."
    c::IntervalSets.ClosedInterval{Float64} = ClosedInterval(1e-4, 1e2)

    "Effective sample size algorithm, used to reweight samples before
    covariance calculation."
    essalg::EAL = EffSampleSizeFromAC()
end

export DynamicMHTuning


mutable struct DynamicMHTunerInstance{
    S<:MCMCBasicStats
} <: AbstractMCMCTunerInstance
    config::DynamicMHTuning
    stats::S
    iteration::Int
    scale::Float64
end

(tuning::DynamicMHTuning)(chain::MHIterator) = DynamicMHTunerInstance(tuning, chain)


function DynamicMHTunerInstance(tuning::DynamicMHTuning, chain::MHIterator)
    m = totalndof(getdensity(chain))
    scale = tuning.s / m
    DynamicMHTunerInstance(tuning, MCMCBasicStats(chain), 1, scale)
end


function tuning_init!(tuner::DynamicMHTunerInstance, chain::MHIterator)
    Σ_unscaled = _approx_cov(getdensity(chain))
    Σ = Σ_unscaled * tuner.scale

    next_cycle!(chain) # ToDo: This would be better placed in the burn-in algorithm
    chain.proposaldist = set_cov(chain.proposaldist, Σ)

    nothing
end


function _append_stats_weighted_by_ess!(stats::MCMCBasicStats, samples::DensitySampleVector)
    if length(eachindex(samples)) >= 1
        n = max(1, minimum(bat_eff_sample_size(unshaped.(samples)).result))
        W = samples.weight .* inv(sum(samples.weight)) .* n
        reweighted_samples = DensitySampleVector((samples.v, samples.logd, W, samples.info, samples.aux))
        append!(stats, samples)
    end
    stats
end


function tuning_postinit!(tuner::DynamicMHTunerInstance, chain::MHIterator, samples::DensitySampleVector)
    # The very first samples of a chain can be very valuable to init tuner
    # stats, especially if the chain gets stuck early after:
    stats = tuner.stats
    _append_stats_weighted_by_ess!(stats, samples)
end


function tuning_update!(tuner::DynamicMHTunerInstance, chain::MHIterator, samples::DensitySampleVector)
    config = tuner.config
    stats = tuner.stats

    stats_reweight_factor = tuner.config.r / tuner.iteration ^ config.λ
    reweight_relative!(stats, stats_reweight_factor)
    _append_stats_weighted_by_ess!(stats, samples)

    α = eff_acceptance_ratio(chain)

    max_log_posterior = stats.logtf_stats.maximum
    if minimum(config.α) <= α <= maximum(config.α)
        chain.info = MCMCIteratorInfo(chain.info, tuned = true)
        @debug "MCMC chain $(chain.info.id) tuned, acceptance ratio = $(Float32(α)), proposal scale = $(Float32(tuner.scale)), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain.info = MCMCIteratorInfo(chain.info, tuned = false)
        @debug "MCMC chain $(chain.info.id) *not* tuned, acceptance ratio = $(Float32(α)), proposal scale = $(Float32(tuner.scale)), max. log posterior = $(Float32(max_log_posterior))"
    end

    c_rescale = proposal_scale_multiplier(α, config.α, config.β, config.γ)
    tuner.scale = clamp(tuner.scale * c_rescale, minimum(config.c), maximum(config.c))

    new_Σ_unscal = convert(Array, stats.param_stats.cov)
    #Σ_old_unscal = Matrix(get_cov(chain.proposaldist)) / c
    #scale_corr_factor = exp(-logabsdet(Σ_old_unscal)[1] + logabsdet(new_Σ_unscal)[1])
    Σ_new = new_Σ_unscal * tuner.scale

    chain.proposaldist = set_cov(chain.proposaldist, Σ_new)
    tuner.iteration += 1

    nothing
end


function proposal_scale_multiplier(α::Real, α_interval::AbstractInterval{<:Real}, β::Real, γ::Real)
    @argcheck minimum(α_interval) > 0
    @argcheck maximum(α_interval) < 1
    @argcheck β >= 1
    @argcheck γ > 0

    R = promote_type(typeof(α), eltype(α_interval), typeof(β), typeof(γ))
    α_min = minimum(α_interval)
    α_max = maximum(α_interval)
    α_mean = mean(α_interval)
    if α ≈ α_mean
        one(R)
    elseif α > α_mean
        h = (α - α_mean) / (1 - α_mean)
        R(1 + h^γ * (β-1))
    elseif α < α_mean
        h = (α_mean - α) / (α_mean - 0)
        R(inv(1 + (h^γ * (β-1))))
    end
end
