# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@with_kw struct RAMTuning <: MCMCTuning
    target_acceptance::Float64 = 0.234 #TODO AC: how to pass custom intitial value for cov matrix?
    σ_target_acceptance::Float64 = 0.05
    gamma::Float64 = 2/3
end
export RAMTuning

mutable struct RAMTrafoTunerState <: AbstractMCMCTunerState
    tuning::RAMTuning
    nsteps::Int
end

mutable struct RAMProposalTunerState <: AbstractMCMCTunerState end

(tuning::RAMTuning)(mc_state::MCMCChainState) = RAMTrafoTunerState(tuning, 0), RAMProposalTunerState()

default_adaptive_transform(tuning::RAMTuning) = TriangularAffineTransform()

RAMTrafoTunerState(tuning::RAMTuning) = RAMTrafoTunerState(tuning, 0)

RAMProposalTunerState(tuning::RAMTuning) = RAMProposalTunerState()

create_trafo_tuner_state(tuning::RAMTuning, chain::MCMCChainState, n_steps_hint::Integer) = RAMTrafoTunerState(tuning, n_steps_hint)

create_proposal_tuner_state(tuning::RAMTuning, chain::MCMCChainState, n_steps_hint::Integer) = RAMProposalTunerState()

function mcmc_tuning_init!!(tuner_state::RAMTrafoTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false) # TODO ?
    tuner_state.nsteps = 0    
    return nothing
end

mcmc_tuning_init!!(tuner_state::RAMProposalTunerState, chain_state::MCMCChainState, max_nsteps::Integer) = nothing

mcmc_tuning_reinit!!(tuner_state::RAMTrafoTunerState, chain_state::MCMCChainState, max_nsteps::Integer) = nothing

mcmc_tuning_reinit!!(tuner_state::RAMProposalTunerState, chain_state::MCMCChainState, max_nsteps::Integer) = nothing

mcmc_tuning_postinit!!(tuner::RAMTrafoTunerState, chain::MCMCChainState, samples::DensitySampleVector) = nothing

mcmc_tuning_postinit!!(tuner::RAMProposalTunerState, chain::MCMCChainState, samples::DensitySampleVector) = nothing


function mcmc_tune_post_cycle!!(tuner::RAMTrafoTunerState, chain::MCMCChainState, samples::DensitySampleVector)
    α_min, α_max = map(op -> op(1, tuner.tuning.σ_target_acceptance), [-,+]) .* tuner.tuning.target_acceptance
    α = eff_acceptance_ratio(chain)

    max_log_posterior = maximum(samples.logd)

    if α_min <= α <= α_max
        chain.info = MCMCChainStateInfo(chain.info, tuned = true)
        @debug "MCMC chain $(chain.info.id) tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain.info = MCMCChainStateInfo(chain.info, tuned = false)
        @debug "MCMC chain $(chain.info.id) *not* tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    end
end

mcmc_tune_post_cycle!!(tuner::RAMProposalTunerState, chain::MCMCChainState, samples::DensitySampleVector) = nothing

mcmc_tuning_finalize!!(tuner::RAMTrafoTunerState, chain::MCMCChainState) = nothing

mcmc_tuning_finalize!!(tuner::RAMProposalTunerState, chain::MCMCChainState) = nothing

tuning_callback(::RAMTrafoTunerState) = nop_func

tuning_callback(::RAMProposalTunerState) = nop_func

# Return mc_state instead of f_transform
function mcmc_tune_post_step!!(
    tuner_state::RAMTrafoTunerState, 
    mc_state::MCMCChainState,
    p_accept::Real,
)
    @unpack target_acceptance, gamma = tuner_state.tuning
    @unpack f_transform, sample_z = mc_state
    
    n_dims = size(sample_z.v[1], 1)
    η = min(1, n_dims * tuner_state.nsteps^(-gamma))

    s_L = f_transform.A

    u = sample_z.v[2] - sample_z.v[1] # proposed - current
    M = s_L * (I + η * (p_accept - target_acceptance) * (u * u') / norm(u)^2 ) * s_L'

    S = cholesky(Positive, M)
    f_transform_new  = Mul(S.L)

    tuner_state.nsteps += 1
    mc_state.f_transform = f_transform_new

    return mc_state, tuner_state, f_transform_new
end

function mcmc_tune_post_step!!(
    tuner_state::RAMProposalTunerState, 
    mc_state::MCMCChainState,
    p_accept::Real,
)
    return mc_state, tuner_state, mc_state.f_transform
end
