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

mcmc_tuning_postinit!!(tuner::RAMTrafoTunerState, chain_state::MCMCChainState, samples::AbstractVector{<:DensitySampleVector}) = nothing


function mcmc_tune_post_cycle!!(
    f_transform::Function,
    tuner::RAMTrafoTunerState, 
    chain_state::MCMCChainState, 
    samples::AbstractVector{<:DensitySampleVector}
)
    α_min = (1 - tuner.tuning.σ_target_acceptance) * tuner.tuning.target_acceptance
    α_max = (1 + tuner.tuning.σ_target_acceptance) * tuner.tuning.target_acceptance
    α = eff_acceptance_ratio(chain_state)

    logds = [walker_smpls.logd for walker_smpls in samples]
    max_log_posterior = maximum(maximum.(logds))

    if α_min <= α <= α_max
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = true)
        @debug "MCMC chain $(chain_state.info.id) tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    else
        chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false)
        @debug "MCMC chain $(chain_state.info.id) *not* tuned, acceptance ratio = $(Float32(α)), max. log posterior = $(Float32(max_log_posterior))"
    end
    return f_transform, tuner, chain_state
end

mcmc_tuning_finalize!!(
    f_transform::Function, 
    tuner::RAMTrafoTunerState, 
    chain::MCMCChainState
) = nothing

function mcmc_tune_post_step!!(
    f_transform::Function,
    tuner_state::RAMTrafoTunerState, 
    chain_state::MCMCChainState,
    current::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    proposed::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    p_accept::AbstractVector{<:Real}
)
    
    if any(current.x.v .== proposed.x.v)
        return f_transform, tuner_state, chain_state
    end
    
    (; target_acceptance, gamma) = tuner_state.tuning
    b = f_transform.b
    n_dims = length(b)

    tuner_state_new = @set tuner_state.nsteps = tuner_state.nsteps + 1

    η = min(1, n_dims * tuner_state.nsteps^(-gamma))

    Σ_L = f_transform.A

    u = proposed.z.v .- current.z.v
    U = stack(u)
    weights = (p_accept .- target_acceptance) ./ norm.(u).^2
    U_w = U .* weights'
    A = Σ_L * (U_w * U') * Σ_L'
    M = Σ_L * Σ_L' + η * A
    Σ_L_new = oftype(Σ_L, cholesky(Positive, M).L)

    mean_update_rate = η / 10 # heuristic
    α = mean_update_rate .* p_accept

    update = α .* (proposed.x.v .- [b])
    new_b = 1 / nwalkers(chain_state) * oftype.(b, sum(update .+ [b])) # = (1 - α) * b + α * proposed.x.v 

    f_transform_new = MulAdd(Σ_L_new, new_b)

    return f_transform_new, tuner_state_new, chain_state
end
