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
    # Statistics of the current adaptation run (since the last geometry
    # commit or cycle start, whichever is more recent), for
    # get_tuning_success:
    run_nobs::Int
    run_accept_sum::Float64
    run_accept_sqsum::Float64
    run_ndivergent::Int
    # Remaining initial observations of the current run excluded from the
    # statistics: right after a dual-averaging (re-)start the step size
    # deliberately overshoots while exploring, routinely producing a
    # divergent trajectory or two that say nothing about the tuned state:
    run_skip::Int
    # Minimum run length (in walker observations) required to consider the
    # step size tuned, derived from the cycle length at (re-)init:
    min_run_nobs::Int
end

function create_proposal_tuner_state(
    tuning::StepSizeAdaptor,
    chain_state::MCMCChainState,
    proposal::HMCProposalState,
    iteration::Integer
)
    log_stepsize = log(proposal.step_size)
    HMCStepSizeTunerState(
        tuning, 0, log_stepsize + log(oftype(log_stepsize, 10)),
        zero(log_stepsize), zero(log_stepsize),
        0, 0.0, 0.0, 0, 0, 50
    )
end

function _restart_dual_averaging!(tuner::HMCStepSizeTunerState, stepsize::Real)
    tuner.log_mu = log(10 * stepsize)
    tuner.m = 0
    tuner.log_stepsize_bar = 0
    tuner.H_bar = 0
    return nothing
end

function _reset_run_stats!(tuner::HMCStepSizeTunerState)
    tuner.run_nobs = 0
    tuner.run_accept_sum = 0.0
    tuner.run_accept_sqsum = 0.0
    tuner.run_ndivergent = 0
    tuner.run_skip = 3
    return nothing
end

function _reset_stepsize_tuner!(tuner::HMCStepSizeTunerState, chain_state::MCMCChainState, max_nsteps::Integer = 0)
    proposal = get_active_proposal(chain_state.proposal)
    if proposal isa HMCProposalState
        _restart_dual_averaging!(tuner, proposal.step_size)
    else
        tuner.m = 0
        tuner.log_stepsize_bar = 0
        tuner.H_bar = 0
    end
    _reset_run_stats!(tuner)
    if max_nsteps > 0
        # The required run length must be satisfiable by the final
        # step-size-only phase of windowed transform tuners (Stan-style
        # term buffers are 50 steps by default, minus the skipped
        # dual-averaging transient):
        tuner.min_run_nobs = min(40, max(20, max_nsteps ÷ 16)) * nwalkers(chain_state)
    end
    return nothing
end

function mcmc_proposal_tuning_init!!(tuner::HMCStepSizeTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    _reset_stepsize_tuner!(tuner, chain_state, max_nsteps)
end

function mcmc_proposal_tuning_reinit!!(tuner::HMCStepSizeTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    _reset_stepsize_tuner!(tuner, chain_state, max_nsteps)
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

# The transform tuner committed a new geometry: find a fresh reasonable
# step size in the new geometry (the pulled-back target and z positions
# are already updated when this is called) and restart dual averaging
# around it. The statistic of the step that crossed the geometry change is
# never seen here, the tuning orchestration discards it:
function mcmc_proposal_transform_committed!!(
    proposal::HMCProposalState,
    tuner::HMCStepSizeTunerState,
    chain_state::MCMCChainState
)
    rng = get_rng(chain_state.context)
    z = chain_state.current.z.v[1]
    stepsize_new = oftype(proposal.step_size, hmc_find_good_stepsize(rng, proposal.target_logdgrad, z))
    proposal_new = @set proposal.step_size = stepsize_new
    _restart_dual_averaging!(tuner, stepsize_new)
    _reset_run_stats!(tuner)
    return proposal_new, tuner, chain_state
end

function mcmc_tune_proposal_post_step!!(
    proposal::HMCProposalState,
    tuner::HMCStepSizeTunerState,
    chain_state::MCMCChainState,
    step_info::MCMCStepInfo
)
    p_accept = step_info.p_accept
    if tuner.run_skip > 0
        tuner.run_skip -= 1
    else
        tuner.run_nobs += length(p_accept)
        tuner.run_accept_sum += sum(p_accept)
        tuner.run_accept_sqsum += sum(abs2, p_accept)
        if !isnothing(step_info.divergent)
            tuner.run_ndivergent += count(step_info.divergent)
        end
    end

    stepsize_new = _dual_averaging_step!(tuner, get_target_acceptance_ratio(proposal), mean(p_accept))
    proposal_new = if isfinite(stepsize_new)
        @set proposal.step_size = oftype(proposal.step_size, stepsize_new)
    else
        proposal
    end

    return proposal_new, tuner, chain_state
end

# HMC tuning success is judged on the statistics the step-size target
# actually refers to (the trajectory acceptance statistic), on trajectory
# health, and on stability - not on the state-movement rate, which is not
# an acceptance probability for multinomial NUTS. Geometry commits during
# a cycle are legitimate (they reset the adaptation run); tuning counts as
# successful once the current run is long enough and healthy, with a
# tolerance calibrated to the statistical noise of the run mean:
function get_tuning_success(
    chain_state::MCMCChainState,
    proposal::HMCProposalState,
    tuner::HMCStepSizeTunerState
)
    n = tuner.run_nobs
    n >= tuner.min_run_nobs || return false
    mean_accept = tuner.run_accept_sum / n
    var_accept = max(tuner.run_accept_sqsum / n - abs2(mean_accept), 0.0)
    se_accept = sqrt(var_accept / n)
    divergent_frac = tuner.run_ndivergent / n
    target = get_target_acceptance_ratio(proposal)
    return abs(mean_accept - target) <= max(0.1, 3 * se_accept) && divergent_frac <= 0.05
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
    # Everything from here on counts as retained sampling in the
    # trajectory diagnostics:
    _mark_warmup_end!(proposal_new.diagnostics)
    return proposal_new, tuner, chain_state
end
