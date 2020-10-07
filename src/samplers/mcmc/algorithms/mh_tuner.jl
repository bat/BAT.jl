# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# ToDo: Add literature references to AdaptiveMHTuning docstring.

"""
    AdaptiveMHTuning(...) <: MHProposalDistTuning

Adaptive MCMC tuning strategy for Metropolis-Hastings samplers.

Adapts the proposal function based on the acceptance ratio and covariance
of the previous samples.

Fields:

* `λ`: Controls the weight given to new covariance information in adapting
  the proposal distribution. Defaults to `0.5`.

* `α`: Metropolis-Hastings acceptance ratio target, tuning will try to
  adapt the proposal distribution to bring the acceptance ratio inside this
  interval. Defaults to `IntervalSets.ClosedInterval(0.15, 0.35)`

* `β`: Controls how much the spread of the proposal distribution is
  widened/narrowed depending on the current MH acceptance ratio.

* `c`: Interval for allowed scale/spread of the proposal distribution.
  Defaults to `ClosedInterval(1e-4, 1e2)`.

* `r`: Reweighting factor. Take accumulated sample statistics of previous
  tuning cycles into account with a relative weight of `r`. Set to `0` to
  completely reset sample statistics between each tuning cycle.

Constructors:

```julia
AdaptiveMHTuning(
    λ::Real,
    α::IntervalSets.ClosedInterval{<:Real},
    β::Real,
    c::IntervalSets.ClosedInterval{<:Real},
    r::Real
)
```
"""
@with_kw struct AdaptiveMHTuning <: MHProposalDistTuning
    λ::Float64 = 0.5
    α::IntervalSets.ClosedInterval{Float64} = ClosedInterval(0.15, 0.35)
    β::Float64 = 1.5
    c::IntervalSets.ClosedInterval{Float64} = ClosedInterval(1e-4, 1e2)
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
    m = totalndof(getdensity(chain))
    scale = 2.38^2 / m
    ProposalCovTuner(tuning, MCMCBasicStats(chain), 1, scale)
end



_approx_cov(target::Distribution) = cov(target)
_approx_cov(target::DistLikeDensity) = cov(target)
_approx_cov(target::AbstractPosteriorDensity) = cov(getprior(target))

function tuning_init!(tuner::ProposalCovTuner, chain::MHIterator)
    Σ_unscaled = _approx_cov(getdensity(chain))
    Σ = Σ_unscaled * tuner.scale

    next_cycle!(chain) # ToDo: This would be better placed in the burn-in algorithm
    chain.proposaldist = set_cov(chain.proposaldist, Σ)

    nothing
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
