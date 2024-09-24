# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@with_kw struct RAMTuning <: MCMCTuning
    target_acceptance::Float64 = 0.234 #TODO AC: how to pass custom intitial value for cov matrix?
    σ_target_acceptance::Float64 = 0.05
    gamma::Float64 = 2/3
end
export RAMTuning

mutable struct RAMTunerState <: AbstractMCMCTunerInstance
    tuning::RAMTuning
    nsteps::Int
end

(tuning::RAMTuning)(mc_state::MCMCState) = RAMTunerState(tuning, 0)

RAMTunerState(tuning::RAMTuning) = RAMTunerState(tuning, 0)

create_tuner_state(tuning::RAMTuning, chain::MCMCState, n_steps_hint::Integer) = RAMTunerState(tuning, n_steps_hint)


function tuning_init!(tuner::RAMTunerState, mc_state::MCMCState, max_nsteps::Integer)
    mc_state.info = MCMCStateInfo(mc_state.info, tuned = false) # TODO ?
    tuner.nsteps = 0
    
    return nothing
end


tuning_postinit!(tuner::RAMTunerState, chain::MCMCState, samples::DensitySampleVector) = nothing

# TODO AC: is this still needed?
# function tuning_postinit!(tuner::ProposalCovTunerState, chain::MCMCState, samples::DensitySampleVector)
#     # The very first samples of a chain can be very valuable to init tuner
#     # stats, especially if the chain gets stuck early after:
#     stats = tuner.stats
#     append!(stats, samples)
# end

tuning_reinit!(tuner::RAMTunerState, chain::MCMCState, max_nsteps::Integer) = nothing


function tuning_update!(tuner::RAMTunerState, chain::MCMCState, samples::DensitySampleVector)
    α_min, α_max = map(op -> op(1, tuner.tuning.σ_target_acceptance), [-,+]) .* tuner.tuning.target_acceptance
    α = eff_acceptance_ratio(chain)

    max_log_posterior = maximum(samples.logd)

    if α_min <= α <= α_max
        chain.info = MCMCStateInfo(chain.info, tuned = true)
        @debug "MCMC chain $(chain.info.id) tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain.info = MCMCStateInfo(chain.info, tuned = false)
        @debug "MCMC chain $(chain.info.id) *not* tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    end
end

tuning_finalize!(tuner::RAMTunerState, chain::MCMCState) = nothing

tuning_callback(::RAMTunerState) = nop_func


default_adaptive_transform(tuner::RAMTuning) = TriangularAffineTransform() 

function tune_transform!!(
    mc_state::MCMCState,
    tuner::RAMTunerState, 
    p_accept::Real,
)
    @unpack target_acceptance, gamma = tuner.tuning
    @unpack f_transform, sample_z = mc_state
    
    n_dims = size(sample_z.v[1], 1)
    η = min(1, n_dims * tuner.nsteps^(-gamma))

    s_L = f_transform.A

    u = sample_z.v[2] - sample_z.v[1] # proposed - current
    M = s_L * (I + η * (p_accept - target_acceptance) * (u * u') / norm(u)^2 ) * s_L'

    S = cholesky(Positive, M)
    f_transform_new  = Mul(S.L)

    tuner.nsteps += 1
    mc_state.f_transform = f_transform_new

    return (tuner, f_transform_new)
end
