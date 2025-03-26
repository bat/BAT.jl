# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# ToDo: Add literature references to AdaptiveAffineTuning docstring.
"""
    struct AdaptiveAffineTuning <: MCMCTransformTuning

Adaptive cycle-based MCMC tuning strategy.

Adapts an affine space transformation based on the acceptance ratio and
covariance of the previous samples.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct AdaptiveAffineTuning <: MCMCTransformTuning
    "Controls the weight given to new covariance information in adapting the
    affine transform."
    λ::Float64 = 0.5

    "Metropolis-Hastings acceptance ratio target, tuning will try to adapt
    the affine transform to bring the acceptance ratio inside this interval."
    α::IntervalSets.ClosedInterval{Float64} = ClosedInterval(0.15, 0.35)

    "Controls how much the scale of the affine transform is
    widened/narrowed depending on the current MH acceptance ratio."
    β::Float64 = 1.5

    "Interval for allowed scale of the affine transform distribution."
    c::IntervalSets.ClosedInterval{Float64} = ClosedInterval(1e-4, 1e2)

    "Reweighting factor. Take accumulated sample statistics of previous
    tuning cycles into account with a relative weight of `r`. Set to
    `0` to completely reset sample statistics between each tuning cycle."
    r::Real = 0.5
end

export AdaptiveAffineTuning

# TODO: MD, make immutable and use Accessors.jl
mutable struct AdaptiveAffineTuningState{
    S<:MCMCBasicStats
} <: MCMCTransformTunerState
    tuning::AdaptiveAffineTuning
    stats::AbstractVector{S}
    iteration::Int
    scale::Float64
end


function AdaptiveAffineTuningState(tuning::AdaptiveAffineTuning, chain_state::MCMCChainState)
    T = eltype(eltype(chain_state.current.x.v))
    scale = one(T)
    AdaptiveAffineTuningState(tuning, MCMCBasicStats(chain_state), 1, scale)
end


create_trafo_tuner_state(tuning::AdaptiveAffineTuning, chain_state::MCMCChainState, iteration::Integer) = AdaptiveAffineTuningState(tuning, chain_state)

mcmc_tuning_init!!(tuner_state::AdaptiveAffineTuningState, chain_state::MCMCChainState, max_nsteps::Integer) = nothing

mcmc_tuning_reinit!!(tuner_state::AdaptiveAffineTuningState, chain_state::MCMCChainState, max_nsteps::Integer) = nothing


function mcmc_tuning_postinit!!(tuner::AdaptiveAffineTuningState, chain_state::MCMCChainState, samples::AbstractVector{<:DensitySampleVector})
    # The very first samples of a chain can be very valuable to init tuner
    # stats, especially if the chain gets stuck early after:
    stats = tuner.stats

    for i in 1:length(samples)
        append!(stats[i], samples[i])
    end
    #map!(append!, stats, samples)
end


# TODO: MD, make properly !!
function mcmc_tune_post_cycle!!(tuner::AdaptiveAffineTuningState, chain_state::MCMCChainState, samples::AbstractVector{<:DensitySampleVector})
    tuning = tuner.tuning
    stats = tuner.stats
    stats_reweight_factor = tuning.r
    reweight_relative!.(stats, stats_reweight_factor)
    n_walkers = length(samples)

    for i in 1:n_walkers
       append!(stats[i], samples[i]) 
    end
    
    #map!(append!, stats, samples)

    α_min = minimum(tuning.α)
    α_max = maximum(tuning.α)

    c_min = minimum(tuning.c)
    c_max = maximum(tuning.c)

    β = tuning.β

    t = tuner.iteration
    λ = tuning.λ
    c = tuner.scale

    f_transform = chain_state.f_transform
    A = f_transform.A
    Σ_old = A * A'

    # TODO, MD: How should the update work for ensembles? Should it be some kind of average?
    param_stats = getfield.(stats, :param_stats)
    covs = convert.(Array, getfield.(param_stats, :cov))
    S = (1/n_walkers) * sum(covs)

    mean_stat = (1/n_walkers) * sum(getfield.(param_stats, :mean))
    
    a_t = 1 / t^λ
    new_Σ_unscal = (1 - a_t) * (Σ_old/c) + a_t * S

    α = eff_acceptance_ratio(chain_state)

    logtf_stats = getfield.(stats, :logtf_stats)
    max_log_posterior = maximum(getfield.(logtf_stats, :maximum)) 

    if α_min <= α <= α_max
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = true)
        @debug "MCMC chain $(chain_state.info.id) tuned, acceptance ratio = $(Float32(α)), proposal scale = $(Float32(c)), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false)
        @debug "MCMC chain $(chain_state.info.id) *not* tuned, acceptance ratio = $(Float32(α)), proposal scale = $(Float32(c)), max. log posterior = $(Float32(max_log_posterior))"

        if α > α_max && c < c_max
            tuner.scale = c * β
        elseif α < α_min && c > c_min
            tuner.scale = c / β
        end
    end

    Σ_new = new_Σ_unscal * tuner.scale
    A_new = oftype(A, cholesky(Positive, Σ_new).L)
    
    # TODO, MD: See comment above. How should the mean update be computed for ensembles? 
    b = chain_state.f_transform.b
    b_new = oftype(b, (1 - a_t) * b + a_t * mean_stat)

    chain_state.f_transform = MulAdd(A_new, b_new)
    
    tuner.iteration += 1

    # TODO: MD, think about keeping old z_position if transform changes only slightly
    chain_state_new = mcmc_update_z_position!!(chain_state) 

    return chain_state_new, tuner
end


mcmc_tuning_finalize!!(tuner::AdaptiveAffineTuningState, chain_state::MCMCChainState) = nothing

# add a boold to return if the transfom changes 
function mcmc_tune_post_step!!(
    tuner::AdaptiveAffineTuningState,
    chain_state::MCMCChainState,
    p_accept::AbstractVector{<:Real}
)
    return chain_state, tuner
end
