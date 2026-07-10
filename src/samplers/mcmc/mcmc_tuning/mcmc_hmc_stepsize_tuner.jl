# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type HMCTuning <: MCMCProposalTuning end


"""
    struct StepSizeAdaptor <: BAT.HMCTuning

Tunes the leapfrog step size of [`HamiltonianMC`](@ref) via Nesterov dual
averaging, targeting the proposal's target acceptance rate.

See Hoffman & Gelman (2014), "The No-U-Turn Sampler: Adaptively Setting Path
Lengths in Hamiltonian Monte Carlo", section 3.2.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct StepSizeAdaptor{T<:Real} <: HMCTuning
    "Adaptation regularization scale."
    gamma::T = 0.05

    "Adaptation iteration offset."
    t0::T = 10.0

    "Adaptation relaxation exponent."
    kappa::T = 0.75
end


mutable struct HMCStepSizeTunerState{T<:AbstractFloat} <: MCMCProposalTunerState
    tuning::StepSizeAdaptor
    m::Int
    log_mu::T
    log_stepsize_bar::T
    H_bar::T
end

function create_proposal_tuner_state(
    tuning::StepSizeAdaptor,
    chain_state::MCMCChainState,
    proposal::HMCProposalState,
    iteration::Integer
)
    log_stepsize = log(proposal.step_size)
    HMCStepSizeTunerState(tuning, 0, log_stepsize + log(oftype(log_stepsize, 10)), zero(log_stepsize), zero(log_stepsize))
end

function _reset_stepsize_tuner!(tuner::HMCStepSizeTunerState, chain_state::MCMCChainState)
    proposal = get_active_proposal(chain_state.proposal)
    if proposal isa HMCProposalState
        tuner.log_mu = log(10 * proposal.step_size)
    end
    tuner.m = 0
    tuner.log_stepsize_bar = 0
    tuner.H_bar = 0
    return nothing
end

function mcmc_proposal_tuning_init!!(tuner::HMCStepSizeTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    _reset_stepsize_tuner!(tuner, chain_state)
end

function mcmc_proposal_tuning_reinit!!(tuner::HMCStepSizeTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    _reset_stepsize_tuner!(tuner, chain_state)
end

# Nesterov dual averaging update, returns the new step size:
function _dual_averaging_step!(tuner::HMCStepSizeTunerState, target_acceptance::Real, alpha::Real)
    (; gamma, t0, kappa) = tuner.tuning
    T = typeof(tuner.H_bar)

    m = tuner.m += 1
    eta_H = 1 / (m + T(t0))
    tuner.H_bar = (1 - eta_H) * tuner.H_bar + eta_H * (T(target_acceptance) - min(one(T), T(alpha)))
    log_stepsize = tuner.log_mu - tuner.H_bar * sqrt(T(m)) / T(gamma)
    eta_x = T(m)^(-T(kappa))
    tuner.log_stepsize_bar = (1 - eta_x) * tuner.log_stepsize_bar + eta_x * log_stepsize

    return exp(log_stepsize)
end

function mcmc_tune_proposal_post_step!!(
    proposal::HMCProposalState,
    tuner::HMCStepSizeTunerState,
    chain_state::MCMCChainState,
    p_accept::AbstractVector{<:Real}
)
    stepsize_new = _dual_averaging_step!(tuner, get_target_acceptance_ratio(proposal), mean(p_accept))
    proposal_new = if isfinite(stepsize_new)
        @set proposal.step_size = oftype(proposal.step_size, stepsize_new)
    else
        proposal
    end

    return proposal_new, tuner, chain_state
end

function mcmc_proposal_tuning_finalize!!(
    proposal::HMCProposalState,
    tuner::HMCStepSizeTunerState,
    chain_state::MCMCChainState
)
    proposal_new = if tuner.m > 0
        @set proposal.step_size = oftype(proposal.step_size, exp(tuner.log_stepsize_bar))
    else
        proposal
    end
    return proposal_new, tuner, chain_state
end
