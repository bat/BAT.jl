@with_kw struct TransformedAdaptiveMHTuning <: MCMCTuningAlgorithm
    "Controls the weight given to new covariance information in adapting the
    proposal distribution."
    λ::Float64 = 0.5

    "Metropolis-Hastings acceptance ratio target, tuning will try to adapt
    the proposal distribution to bring the acceptance ratio inside this interval."
    α::IntervalSets.ClosedInterval{Float64} = ClosedInterval(0.15, 0.35)

    "Controls how much the spread of the proposal distribution is
    widened/narrowed depending on the current MH acceptance ratio."
    β::Float64 = 1.5

    "Interval for allowed scale/spread of the proposal distribution."
    c::IntervalSets.ClosedInterval{Float64} = ClosedInterval(1e-4, 1e2)

    "Reweighting factor. Take accumulated sample statistics of previous
    tuning cycles into account with a relative weight of `r`. Set to
    `0` to completely reset sample statistics between each tuning cycle."
    r::Real = 0.5
end

mutable struct TransformedProposalCovTuner{
    S<:MCMCBasicStats
} <: AbstractMCMCTunerInstance
    config::TransformedAdaptiveMHTuning
    stats::S
    iteration::Int
    scale::Float64
end


function TransformedProposalCovTuner(tuning::TransformedAdaptiveMHTuning, chain::MCMCIterator)
    m = totalndof(varshape(getmeasure(chain)))
    scale = 2.38^2 / m
    TransformedProposalCovTuner(tuning, MCMCBasicStats(chain), 1, scale)
end

get_tuner(tuning::TransformedAdaptiveMHTuning, chain::MCMCIterator) =  TransformedProposalCovTuner(tuning, chain)
default_adaptive_transform(tuner::TransformedAdaptiveMHTuning) = TriangularAffineTransform() 
default_adaptive_transform(algorithm::MetropolisHastings) = TriangularAffineTransform() 

function tuning_init!(tuner::TransformedProposalCovTuner, chain::MCMCIterator, max_nsteps::Integer)
    chain.info = MCMCIteratorInfo(chain.info, tuned = false)

    nothing
end

tuning_reinit!(tuner::TransformedProposalCovTuner, chain::MCMCIterator, max_nsteps::Integer) = nothing


function tuning_postinit!(tuner::TransformedProposalCovTuner, chain::MCMCIterator, samples::DensitySampleVector)
    # The very first samples of a chain can be very valuable to init tuner
    # stats, especially if the chain gets stuck early after:
    stats = tuner.stats
    append!(stats, samples)
end

# this function is called once after each tuning cycle
g_state = nothing
function tuning_update!(tuner::TransformedProposalCovTuner, chain::TransformedMCMCIterator, samples::DensitySampleVector)
    
    stats = tuner.stats
    stats_reweight_factor = tuner.config.r
    reweight_relative!(stats, stats_reweight_factor)
    # empty!.(stats)
    append!(stats, samples)


    config = tuner.config

    α_min = minimum(config.α)
    α_max = maximum(config.α)

    c_min = minimum(config.c)
    c_max = maximum(config.c)

    β = config.β

    t = tuner.iteration
    λ = config.λ
    c = tuner.scale
    
    transform = chain.f_transform

    A = transform.A
    Σ_old = A*A'

    S = convert(Array, stats.param_stats.cov)
    a_t = 1 / t^λ
    new_Σ_unscal = (1 - a_t) * (Σ_old/c) + a_t * S

    α = eff_acceptance_ratio(chain)

    max_log_posterior = stats.logtf_stats.maximum

    if α_min <= α <= α_max
        chain.info = MCMCIteratorInfo(chain.info, tuned = true)
        @debug "MCMC chain $(chain.info.id) tuned, acceptance ratio = $(Float32(α)), proposal scale = $(Float32(c)), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain.info = MCMCIteratorInfo(chain.info, tuned = false)
        @debug "MCMC chain $(chain.info.id) *not* tuned, acceptance ratio = $(Float32(α)), proposal scale = $(Float32(c)), max. log posterior = $(Float32(max_log_posterior))"

        if α > α_max && c < c_max
            tuner.scale = c * β
        elseif α < α_min && c > c_min
            tuner.scale = c / β
        end
    end

    Σ_new = new_Σ_unscal * tuner.scale

    S_new = cholesky(Positive, Σ_new)
    chain.f_transform = Mul(S_new.L)
    tuner.iteration += 1

    nothing
    
end


tuning_finalize!(tuner::TransformedProposalCovTuner, chain::MCMCIterator) = nothing

tuning_callback(::TransformedProposalCovTuner) = nop_func

# default_adaptive_transform(tuner::TransformedProposalCovTuner) = TriangularAffineTransform() 


# this function is called in each mcmc_iterate step during tuning 
function tune_mcmc_transform!!(
    tuner::TransformedProposalCovTuner,
    transform::Any, #AffineMaps.AbstractAffineMap,#{<:typeof(*), <:LowerTriangular{<:Real}},
    p_accept::Real,
    z_proposed::Vector{<:Float64}, #TODO: use DensitySamples instead
    z_current::Vector{<:Float64},
    stepno::Int,
    context::BATContext
)

    return (tuner, transform, false)
end

