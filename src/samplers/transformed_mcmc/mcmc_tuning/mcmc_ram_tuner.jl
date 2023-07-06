@with_kw struct TransformedRAMTuner <: TransformedMCMCTuningAlgorithm #TODO: rename to RAMTuning
    target_acceptance::Float64 = 0.234 #TODO AC: how to pass custom intitial value for cov matrix?
    σ_target_acceptance::Float64 = 0.05
    gamma::Float64 = 2/3
end

@with_kw mutable struct TransformedRAMTunerInstance <: TransformedAbstractMCMCTunerInstance
    config::TransformedRAMTuner
    nsteps::Int = 0
end
TransformedRAMTunerInstance(ram::TransformedRAMTuner) = TransformedRAMTunerInstance(config = ram)

get_tuner(tuning::TransformedRAMTuner, chain::MCMCIterator) = TransformedRAMTunerInstance(tuning) 


function tuning_init!(tuner::TransformedRAMTunerInstance, chain::MCMCIterator, max_nsteps::Integer)
    chain.info = TransformedMCMCIteratorInfo(chain.info, tuned = false) # TODO ?
    tuner.nsteps = 0
    
    return nothing
end


tuning_postinit!(tuner::TransformedRAMTunerInstance, chain::MCMCIterator, samples::DensitySampleVector) = nothing

# TODO AC: is this still needed?
# function tuning_postinit!(tuner::TransformedProposalCovTuner, chain::MCMCIterator, samples::DensitySampleVector)
#     # The very first samples of a chain can be very valuable to init tuner
#     # stats, especially if the chain gets stuck early after:
#     stats = tuner.stats
#     append!(stats, samples)
# end

tuning_reinit!(tuner::TransformedRAMTunerInstance, chain::MCMCIterator, max_nsteps::Integer) = nothing





function tuning_update!(tuner::TransformedRAMTunerInstance, chain::MCMCIterator, samples::DensitySampleVector)
    α_min, α_max = map(op -> op(1, tuner.config.σ_target_acceptance), [-,+]) .* tuner.config.target_acceptance
    α = eff_acceptance_ratio(chain)

    max_log_posterior = maximum(samples.logd)

    if α_min <= α <= α_max
        chain.info = TransformedMCMCIteratorInfo(chain.info, tuned = true)
        @debug "MCMC chain $(chain.info.id) tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain.info = TransformedMCMCIteratorInfo(chain.info, tuned = false)
        @debug "MCMC chain $(chain.info.id) *not* tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    end
end

tuning_finalize!(tuner::TransformedRAMTunerInstance, chain::MCMCIterator) = nothing

# tuning_callback(::TransformedRAMTuner) = nop_func



default_adaptive_transform(tuner::TransformedRAMTuner) = TriangularAffineTransform() 

function tune_mcmc_transform!!(
    tuner::TransformedRAMTunerInstance, 
    transform::Mul{<:LowerTriangular}, #AffineMaps.AbstractAffineMap,#{<:typeof(*), <:LowerTriangular{<:Real}},
    p_accept::Real,
    z_proposed::Vector{<:Float64}, #TODO: use DensitySamples instead
    z_current::Vector{<:Float64},
    stepno::Int,
    context::BATContext
)
    @unpack target_acceptance, gamma = tuner.config
    n = size(z_current,1)
    η = min(1, n * stepno^(-gamma))

    s_L = transform.A

    u = z_proposed-z_current
    M = s_L * (I + η * (p_accept - target_acceptance) * (u * u') / norm(u)^2 ) * s_L'

    S = cholesky(Positive, M)
    transform_new  = Mul(S.L)

    tuner.nsteps += 1

    return (tuner, transform_new)
end
