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

mutable struct ProposalCovTunerState{
    S<:MCMCBasicStats
} <: AbstractMCMCTunerInstance
    tuning::AdaptiveMHTuning
    stats::S
    iteration::Int
    scale::Float64
end

(tuning::AdaptiveMHTuning)(mc_state::MCMCState) = ProposalCovTunerState(tuning, mc_state)

# TODO: MD, what should the default be? 
default_adaptive_transform(tuning::AdaptiveMHTuning) = TriangularAffineTransform()

function ProposalCovTunerState(tuning::AdaptiveMHTuning, mc_state::MCMCState)
    m = totalndof(varshape(mcmc_target(mc_state)))
    scale = 2.38^2 / m
    ProposalCovTunerState(tuning, MCMCBasicStats(mc_state), 1, scale)
end


function _cov_with_fallback(d::UnivariateDistribution, n::Integer)
    rng = _bat_determ_rng()
    T = float(eltype(rand(rng, d)))
    C = fill(T(NaN), n, n)
    try
        C[:] = Diagonal(fill(var(d),n))
    catch err
        if err isa MethodError
            C[:] = Diagonal(fill(var(nestedview(rand(rng, d, 10^5))),n))
        else
            throw(err)
        end
    end
    return C
end

function _cov_with_fallback(d::MultivariateDistribution, n::Integer)
    rng = _bat_determ_rng()
    T = float(eltype(rand(rng, d)))
    C = fill(T(NaN), n, n)
    try
        C[:] = cov(d)
    catch err
        if err isa MethodError
            C[:] = cov(nestedview(rand(rng, d, 10^5)))
        else
            throw(err)
        end
    end
    return C
end

_approx_cov(target::Distribution, n) = _cov_with_fallback(target, n)
_approx_cov(target::BATDistMeasure, n) = _cov_with_fallback(Distribution(target), n)
_approx_cov(target::AbstractPosteriorMeasure, n) = _approx_cov(getprior(target), n)

function tuning_init!(tuner::ProposalCovTunerState, mc_state::MCMCState, max_nsteps::Integer)
    n = totalndof(varshape(mcmc_target(mc_state)))

    proposaldist = mc_state.proposal.proposaldist
    Σ_unscaled = _approx_cov(proposaldist, n)
    Σ = Σ_unscaled * tuner.scale
    
    proposaldist = set_cov(proposaldist, Σ)

    nothing
end


tuning_reinit!(tuner::ProposalCovTunerState, mc_state::MCMCState, max_nsteps::Integer) = nothing


function tuning_postinit!(tuner::ProposalCovTunerState, mc_state::MHState, samples::DensitySampleVector)
    # The very first samples of a chain can be very valuable to init tuner
    # stats, especially if the chain gets stuck early after:
    stats = tuner.stats
    append!(stats, samples)
end


function tuning_update!(tuner::ProposalCovTunerState, mc_state::MHState, samples::DensitySampleVector)
    tuning = tuner.tuning
    stats = tuner.stats
    stats_reweight_factor = tuning.r
    reweight_relative!(stats, stats_reweight_factor)
    append!(stats, samples)

    proposaldist = mc_state.proposal.proposaldist

    α_min = minimum(tuning.α)
    α_max = maximum(tuning.α)

    c_min = minimum(tuning.c)
    c_max = maximum(tuning.c)

    β = tuning.β

    t = tuner.iteration
    λ = tuning.λ
    c = tuner.scale

    f_transform = mc_state.f_transform
    A = f_transform.A
    Σ_old = A * A'

    S = convert(Array, stats.param_stats.cov)
    a_t = 1 / t^λ
    new_Σ_unscal = (1 - a_t) * (Σ_old/c) + a_t * S

    α = eff_acceptance_ratio(mc_state)

    max_log_posterior = stats.logtf_stats.maximum

    if α_min <= α <= α_max
        mc_state.info = MCMCStateInfo(mc_state.info, tuned = true)
        @debug "MCMC chain $(mc_state.info.id) tuned, acceptance ratio = $(Float32(α)), proposal scale = $(Float32(c)), max. log posterior = $(Float32(max_log_posterior))"
    else
        mc_state.info = MCMCStateInfo(mc_state.info, tuned = false)
        @debug "MCMC chain $(mc_state.info.id) *not* tuned, acceptance ratio = $(Float32(α)), proposal scale = $(Float32(c)), max. log posterior = $(Float32(max_log_posterior))"

        if α > α_max && c < c_max
            tuner.scale = c * β
        elseif α < α_min && c > c_min
            tuner.scale = c / β
        end
    end

    Σ_new = new_Σ_unscal * tuner.scale
    S_new = cholesky(Positive, Σ_new)
    
    mc_state.f_transform = Mul(S_new.L)
    
    tuner.iteration += 1

    nothing
end

tuning_finalize!(tuner::ProposalCovTunerState, mc_state::MCMCState) = nothing

tuning_callback(::ProposalCovTunerState) = nop_func

tuning_callback(::Nothing) = nop_func

function tune_transform!!(
    mc_state::MCMCState,
    tuner::ProposalCovTunerState,
    p_accept::Real
)
    return (tuner, mc_state.f_transform)
end

function tune_transform!!(
    mc_state::MCMCState,
    tuner::Nothing,
    p_accept::Real
)
    return (tuner, mc_state.f_transform)
end
