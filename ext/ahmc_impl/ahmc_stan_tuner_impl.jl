# This file is a part of BAT.jl, licensed under the MIT License (MIT).

mutable struct StanLikeTunerState{
    S<:MCMCBasicStats,
} <: MCMCTransformTunerState
    tuning::StanLikeTuning
    target_acceptance::Float64
    stats::S
    stan_state::AdvancedHMC.Adaptation.StanHMCAdaptorState
end

BAT.create_trafo_tuner_state(tuning::StanLikeTuning, chain_state::MCMCChainState, n_steps_hint::Integer) = StanLikeTunerState(tuning, tuning.target_acceptance, MCMCBasicStats(chain_state), AdvancedHMC.Adaptation.StanHMCAdaptorState())

function BAT.mcmc_tuning_init!!(tuner::StanLikeTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    tuning = tuner.tuning
    AdvancedHMC.Adaptation.initialize!(tuner.stan_state, tuning.init_buffer, tuning.term_buffer, tuning.window_size, Int(max_nsteps - 1))
    nothing
end

function BAT.mcmc_tuning_reinit!!(tuner::StanLikeTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    tuning = tuner.tuning
    AdvancedHMC.Adaptation.initialize!(tuner.stan_state, tuning.init_buffer, tuning.term_buffer, tuning.window_size, Int(max_nsteps - 1))
    nothing
end

BAT.mcmc_tuning_postinit!!(tuner::StanLikeTunerState, chain_state::MCMCChainState, samples::AbstractVector{<:DensitySampleVector}) = nothing


function BAT.mcmc_tune_post_cycle!!(
    f_transform::Function,
    tuner::StanLikeTunerState,
    chain_state::MCMCChainState,
    proposal::MCMCProposalState,
    samples::AbstractVector{<:DensitySampleVector}
)
    logds = [walker_smpls.logd for walker_smpls in samples]
    max_log_posterior = maximum(maximum.(logds))
    accept_ratio = eff_acceptance_ratio(chain_state) 
    α_min, _ = get_target_accept_interval(proposal)
    if accept_ratio >= α_min
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = true)
        @debug "MCMC chain $(chain_state.info.id) tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain_state.proposal.τ.integrator), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false)
        @debug "MCMC chain $(chain_state.info.id) *not* tuned, acceptance ratio = $(Float32(accept_ratio)), integrator = $(chain_state.proposal.τ.integrator), max. log posterior = $(Float32(max_log_posterior))"
    end
    return f_transform, tuner, chain_state
end


BAT.mcmc_tuning_finalize!!(f_transform::Function, tuner::StanLikeTunerState, chain_state::MCMCChainState) = nothing


function BAT.mcmc_tune_post_step!!(
    f_transform::Function,
    tuner::StanLikeTunerState,
    chain_state::MCMCChainState,
    proposal::MCMCProposalState,
    current::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    proposed::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    p_accept::AbstractVector{<:Real}
)
    stan_state = tuner.stan_state
    stan_state.i += 1

    stats = tuner.stats
    is_in_window =  stan_state.i >= stan_state.window_start && stan_state.i <= stan_state.window_end
    is_window_end = stan_state.i in stan_state.window_splits

    f_transform_new = deepcopy(f_transform)

    if is_in_window
        BAT.append!(stats, chain_state.proposed.x)
    end
    
    if is_window_end 
        A = f_transform.A
        T = eltype(A)
        n_dims = size(A, 2)

        M = convert(Array, stats.param_stats.cov)

        A_new = T.(cholesky(Positive, M).L)

        reweight_relative!(stats, 0)

        f_transform_new = MulAdd(A_new, zeros(T, n_dims))
    end

    return f_transform_new, tuner, chain_state
end
