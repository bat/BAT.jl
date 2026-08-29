# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Scalar Langevin step-scale adaptation for MALA via Nesterov dual
# averaging (see StepSizeAdaptor): the transform tuner learns the target
# geometry, the step-scale tuner steers the acceptance rate towards the
# proposal's target - the same separation of responsibilities as for HMC
# (transform + leapfrog step size). Without it, the fixed dimension-based
# τ heuristic can leave the acceptance rate outside the target interval
# on non-Gaussian targets, with no mechanism to correct it.

mutable struct MALAStepSizeTunerState{T<:AbstractFloat} <: DualAveragingTunerState
    tuning::StepSizeAdaptor
    m::Int
    log_mu::T
    log_stepsize_bar::T
    H_bar::T
    run_nobs::Int
    run_accept_sum::Float64
    min_run_nobs::Int
end

function create_proposal_tuner_state(
    tuning::StepSizeAdaptor,
    chain_state::MCMCChainState,
    proposal::MALAProposalState,
    iteration::Integer
)
    _validate_dual_averaging_domain(tuning)
    log_tau = log(proposal.τ)
    MALAStepSizeTunerState(
        tuning, 0, log_tau + log(oftype(log_tau, 10)),
        zero(log_tau), zero(log_tau), 0, 0.0, 50
    )
end

function _reset_mala_stepsize_tuner!(
    tuner::MALAStepSizeTunerState,
    chain_state::MCMCChainState,
)
    proposal = get_active_proposal(chain_state.proposal)
    if proposal isa MALAProposalState
        _restart_dual_averaging!(tuner, proposal.τ)
    else
        tuner.m = 0
        tuner.log_stepsize_bar = 0
        tuner.H_bar = 0
    end
    tuner.run_nobs = 0
    tuner.run_accept_sum = zero(tuner.run_accept_sum)
    tuner.min_run_nobs = 0
    return nothing
end

function mcmc_proposal_tuning_init!!(tuner::MALAStepSizeTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    _reset_mala_stepsize_tuner!(tuner, chain_state)
end

function mcmc_proposal_tuning_reinit!!(tuner::MALAStepSizeTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    _reset_mala_stepsize_tuner!(tuner, chain_state)
end

# The transform tuner committed a new geometry: the current τ remains the
# best starting point (the dimension-based default is already
# geometry-normalized), but the acceptance statistics gathered under the
# old geometry no longer apply, so dual averaging restarts around it:
function mcmc_proposal_transform_committed!!(
    proposal::MALAProposalState,
    tuner::MALAStepSizeTunerState,
    chain_state::MCMCChainState
)
    _reset_mala_stepsize_tuner!(tuner, chain_state)
    return proposal, tuner, chain_state
end

function mcmc_tune_proposal_post_step!!(
    proposal::MALAProposalState,
    tuner::MALAStepSizeTunerState,
    chain_state::MCMCChainState,
    step_info::MCMCStepInfo
)
    p_accept = step_info.p_accept
    accept_sum = _ordered_walker_sum(p_accept, step_info.walker_order)
    mean_accept = accept_sum / length(p_accept)
    tau_new = _dual_averaging_step!(
        tuner, get_target_acceptance_ratio(proposal), mean_accept,
    )
    tuner.run_nobs += length(p_accept)
    tuner.run_accept_sum += accept_sum
    proposal_new = @set proposal.τ = oftype(proposal.τ, tau_new)
    return proposal_new, tuner, chain_state
end

function get_tuning_success(
    chain_state::MCMCChainState,
    proposal::MALAProposalState,
    tuner::MALAStepSizeTunerState,
)
    tuner.min_run_nobs == 0 && return get_tuning_success(chain_state, proposal)
    tuner.run_nobs >= tuner.min_run_nobs || return false
    acceptance = tuner.run_accept_sum / tuner.run_nobs
    lower, upper = get_target_acceptance_int(proposal)
    return lower <= acceptance <= upper
end

function mcmc_proposal_tuning_finalize!!(
    proposal::MALAProposalState,
    tuner::MALAStepSizeTunerState,
    chain_state::MCMCChainState
)
    proposal_new = if tuner.m > 0
        @set proposal.τ = oftype(proposal.τ, _dual_averaging_final_scale(tuner))
    else
        proposal
    end
    return proposal_new, tuner, chain_state
end
