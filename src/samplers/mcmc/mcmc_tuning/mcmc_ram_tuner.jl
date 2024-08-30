@with_kw struct RAMTuning <: MCMCTuningAlgorithm #TODO: rename to RAMTuning
    target_acceptance::Float64 = 0.234 #TODO AC: how to pass custom intitial value for cov matrix?
    σ_target_acceptance::Float64 = 0.05
    gamma::Float64 = 2/3
end

mutable struct RAMTunerState <: AbstractMCMCTunerInstance
    tuning::RAMTuning
    nsteps::Int
end
RAMTunerState(ram::RAMTuning) = RAMTunerState(tuning = ram, nsteps = 0)

get_tuner(tuning::RAMTuning, chain::MCMCState) = RAMTunerState(tuning)# TODO rename to create_tuner_state(tuning::RAMTuner, mc_state::MCMCState, n_steps_hint::Integer)


function tuning_init!(tuner::RAMTunerState, chain::MCMCState, max_nsteps::Integer)
    chain.info = MCMCStateInfo(chain.info, tuned = false) # TODO ?
    tuner.nsteps = 0
    
    return nothing
end


tuning_postinit!(tuner::RAMTunerState, chain::MCMCState, samples::DensitySampleVector) = nothing

# TODO AC: is this still needed?
# function tuning_postinit!(tuner::ProposalCovTuner, chain::MCMCState, samples::DensitySampleVector)
#     # The very first samples of a chain can be very valuable to init tuner
#     # stats, especially if the chain gets stuck early after:
#     stats = tuner.stats
#     append!(stats, samples)
# end

tuning_reinit!(tuner::RAMTunerState, chain::MCMCState, max_nsteps::Integer) = nothing





function tuning_update!(tuner::RAMTunerState, chain::MCMCState, samples::DensitySampleVector)
    α_min, α_max = map(op -> op(1, tuner.config.σ_target_acceptance), [-,+]) .* tuner.config.target_acceptance
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

# tuning_callback(::RAMTuning) = nop_func



default_adaptive_transform(tuner::RAMTuning) = TriangularAffineTransform() 

function tune_mcmc_transform!!(
    tuner::RAMTunerState, 
    transform::Mul{<:LowerTriangular}, #AffineMaps.AbstractAffineMap,#{<:typeof(*), <:LowerTriangular{<:Real}},
    p_accept::Real,
    sample_z,
    stepno::Int,
    context::BATContext
)
    @unpack target_acceptance, gamma = tuner.config
    n = size(sample_z.v[1],1)
    η = min(1, n * tuner.nsteps^(-gamma))

    s_L = transform.A

    u = sample_z.v[2] - sample_z.v[1] # proposed - current
    M = s_L * (I + η * (p_accept - target_acceptance) * (u * u') / norm(u)^2 ) * s_L'

    S = cholesky(Positive, M)
    transform_new  = Mul(S.L)

    tuner.nsteps += 1

    return (tuner, transform_new, true)
end
