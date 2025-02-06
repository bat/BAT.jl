# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct RAMTuning <: MCMCTransformTuning

Tunes MCMC spaces transformations based on the
[Robust adaptive Metropolis algorithm](https://doi.org/10.1007/s11222-011-9269-5).

In constrast to the original RAM algorithm, `RAMTuning` does not use the
covariance estimate to change a proposal distribution, but instead
uses it as the bases for an affine transformation. The sampling process is
mathematically equivalent, though.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct RAMTuning <: MCMCTransformTuning
    "MCMC target acceptance ratio."
    target_acceptance::Float64 = 0.234

    "Width around `target_acceptance`."
    σ_target_acceptance::Float64 = 0.05

    "Negative adaption rate exponent."
    gamma::Float64 = 2/3
end
export RAMTuning

mutable struct RAMTrafoTunerState <: MCMCTransformTunerState
    tuning::RAMTuning
    nsteps::Int
end

mutable struct RAMProposalTunerState <: MCMCTransformTunerState end


create_trafo_tuner_state(tuning::RAMTuning, chain_state::MCMCChainState, n_steps_hint::Integer) = RAMTrafoTunerState(tuning, 0)

function mcmc_tuning_init!!(tuner_state::RAMTrafoTunerState, chain_state::MCMCChainState, max_nsteps::Integer)
    chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false) # TODO ?
    tuner_state.nsteps = 0    
    return nothing
end

mcmc_tuning_reinit!!(tuner_state::RAMTrafoTunerState, chain_state::MCMCChainState, max_nsteps::Integer) = nothing

mcmc_tuning_postinit!!(tuner::RAMTrafoTunerState, chain::MCMCChainState, samples::DensitySampleVector) = nothing


function mcmc_tune_post_cycle!!(tuner::RAMTrafoTunerState, chain_state::MCMCChainState, samples::DensitySampleVector)
    α_min = (1 - tuner.tuning.σ_target_acceptance) * tuner.tuning.target_acceptance
    α_max = (1 + tuner.tuning.σ_target_acceptance) * tuner.tuning.target_acceptance
    α = eff_acceptance_ratio(chain_state)

    max_log_posterior = maximum(samples.logd)

    if α_min <= α <= α_max
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = true)
        @debug "MCMC chain $(chain_state.info.id) tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false)
        @debug "MCMC chain $(chain_state.info.id) *not* tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    end
    return chain_state, tuner
end

mcmc_tuning_finalize!!(tuner::RAMTrafoTunerState, chain::MCMCChainState) = nothing

function mcmc_tune_post_step!!(
    tuner_state::RAMTrafoTunerState, 
    mc_state::MCMCChainState,
    p_accept::Real,
)
    # TODO: MD: Discuss; apparently the RandomWalk sampler wants the trafo to be tuned even if p_accept = 0. If not, the burnin does not converge.
    if iszero(p_accept) && !(mc_state isa MHChainState)
        return mc_state, tuner_state
    end

    (; f_transform, sample_z) = mc_state
    (; target_acceptance, gamma) = tuner_state.tuning
    b = f_transform.b
    
    tuner_state_new = @set tuner_state.nsteps = tuner_state.nsteps + 1
    
    n_dims = size(sample_z.v[1], 1)
    η = min(1, n_dims * tuner_state.nsteps^(-gamma))

    s_L = f_transform.A

    u = sample_z.v[2] - sample_z.v[1] # proposed - current
    M = s_L * (I + η * (p_accept - target_acceptance) * (u * u') / norm(u)^2 ) * s_L'

    new_s_L = oftype(s_L, cholesky(Positive, M).L)
    
    x = mc_state.samples[_proposed_sample_idx(mc_state)] # proposed in x-space
    mean_update_rate = η / 10 # heuristic
    α = mean_update_rate * p_accept
    new_b = oftype(b, (1- α) * b + α * x.v)

    f_transform_new = MulAdd(new_s_L, new_b)
    
    mc_state_new = set_mc_state_transform!!(mc_state, f_transform_new)
    mc_state_new = mcmc_update_z_position!!(mc_state_new)

    return mc_state_new, tuner_state_new
end
