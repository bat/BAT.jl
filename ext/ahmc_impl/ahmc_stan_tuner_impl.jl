# This file is a part of BAT.jl, licensed under the MIT License (MIT).

mutable struct StanHMCTrafoTunerState{
    S<:MCMCBasicStats,
} <: MCMCTransformTunerState
    tuning::StanHMCTuning
    target_acceptance::Float64
    stats::S
    stan_state::AdvancedHMC.Adaptation.StanHMCAdaptorState
end

BAT.create_trafo_tuner_state(tuning::StanHMCTuning, chain_state::MCMCChainState, n_steps_hint::Integer) = StanHMCTrafoTunerState(tuning, tuning.target_acceptance, MCMCBasicStats(chain_state), AdvancedHMC.Adaptation.StanHMCAdaptorState())

function BAT.mcmc_tuning_init!!(tuner::StanHMCTrafoTunerState, chain_state::HMCState, max_nsteps::Integer)
    tuning = tuner.tuning
    AdvancedHMC.Adaptation.initialize!(tuner.stan_state, tuning.init_buffer, tuning.term_buffer, tuning.window_size, Int(max_nsteps - 1))
    nothing
end

function BAT.mcmc_tuning_reinit!!(tuner::StanHMCTrafoTunerState, chain_state::HMCState, max_nsteps::Integer)
    tuning = tuner.tuning
    AdvancedHMC.Adaptation.initialize!(tuner.stan_state, tuning.init_buffer, tuning.term_buffer, tuning.window_size, Int(max_nsteps - 1))
    nothing
end

BAT.mcmc_tuning_postinit!!(tuner::StanHMCTrafoTunerState, chain_state::HMCState, samples::DensitySampleVector) = nothing


function BAT.mcmc_tune_post_cycle!!(tuner::StanHMCTrafoTunerState, chain_state::HMCState, samples::DensitySampleVector)
    max_log_posterior = maximum(samples.logd)
    accept_ratio = eff_acceptance_ratio(chain_state)
    if accept_ratio >= 0.9 * tuner.target_acceptance
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = true)
        @debug "MCMC chain $(chain_state.info.id) tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain_state.proposal.τ.integrator), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false)
        @debug "MCMC chain $(chain_state.info.id) *not* tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain_state.proposal.τ.integrator), max. log posterior = $(Float32(max_log_posterior))"
    end
    return chain_state, tuner
end


BAT.mcmc_tuning_finalize!!(tuner::StanHMCTrafoTunerState, chain_state::HMCState) = nothing


function BAT.mcmc_tune_post_step!!(
    tuner::StanHMCTrafoTunerState,
    chain_state::MCMCChainState,
    p_accept::Real
)
    stan_state = tuner.stan_state
    stan_state.i += 1

    stats = tuner.stats
    is_in_window =  stan_state.i >= stan_state.window_start && stan_state.i <= stan_state.window_end
    is_window_end = stan_state.i in stan_state.window_splits

    if is_in_window
        BAT.push!(stats, proposed_sample(chain_state))
    end

    if is_window_end 
        A = chain_state.f_transform.A
        T = eltype(A)
        n_dims = size(A, 2)
        
        M = convert(Array, stats.param_stats.cov)
        A_new = T.(cholesky(Positive, M).L)

        reweight_relative!(stats, 0)

        f_transform_new = MulAdd(A_new, zeros(T, n_dims))
        chain_state = set_mc_state_transform!!(chain_state, f_transform_new)
    end

    chain_state_new = mcmc_update_z_position!!(chain_state) 

    return chain_state_new, tuner
end
