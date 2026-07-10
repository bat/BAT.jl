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
    "Negative adaption rate exponent."
    gamma::Float64 = 2/3
end
export RAMTuning

mutable struct RAMTrafoTunerState <: MCMCTransformTunerState
    tuning::RAMTuning
    nsteps::Int
end

mutable struct RAMProposalTunerState <: MCMCTransformTunerState end


create_trafo_tuner_state(
    tuning::RAMTuning,
    chain_state::MCMCChainState,
    n_steps_hint::Integer
) = RAMTrafoTunerState(tuning, 0)

function mcmc_trafo_tuning_init!!(
    tuner_state::RAMTrafoTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false) # TODO ?
    tuner_state.nsteps = 0
    return nothing
end

# Computes the lower Cholesky factor of `L * (I + sum_i w_i * u_i * u_iᵀ) * Lᵀ`
# via rank-one factor updates in O(k n²) instead of O(n³), following the
# original RAM formulation (Vihola 2012). Falls back to a full modified
# Cholesky decomposition if `L` is not triangular or rounding errors break
# positive definiteness.
function _rank_k_cholesky_update(L::AbstractMatrix{<:Real}, u::AbstractVector{<:AbstractVector{<:Real}}, w::AbstractVector{<:Real})
    if istril(L)
        C = Cholesky(LowerTriangular(Matrix(L)))
        try
            for i in eachindex(u, w)
                v = sqrt(abs(w[i])) .* (L * u[i])
                w[i] >= 0 ? lowrankupdate!(C, v) : lowrankdowndate!(C, v)
            end
            return C.L
        catch err
            err isa PosDefException || rethrow()
        end
    end
    M = L * (I + sum(w[i] .* u[i] .* u[i]' for i in eachindex(u, w))) * L'
    return cholesky(Positive, M).L
end

function mcmc_tune_trafo_post_step!!(
    f_transform::Function,
    tuner_state::RAMTrafoTunerState,
    chain_state::MCMCChainState,
    proposal::MCMCProposalState,
    current::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    proposed::NamedTuple{<:Any, <:Tuple{Vararg{DensitySampleVector}}},
    p_accept::AbstractVector{<:Real}
)

    if any(current.x.v .== proposed.x.v)
        return f_transform, tuner_state, chain_state
    end

    gamma = tuner_state.tuning.gamma
    target_acceptance = get_target_acceptance_ratio(proposal)
    b = f_transform.b
    n_dims = length(b)

    tuner_state_new = @set tuner_state.nsteps = tuner_state.nsteps + 1

    η = min(1, n_dims * tuner_state.nsteps^(-gamma))

    Σ_L = f_transform.A

    u = proposed.z.v .- current.z.v
    weights = (p_accept .- target_acceptance) ./ norm.(u).^2
    Σ_L_new = oftype(Σ_L, _rank_k_cholesky_update(Σ_L, u, η .* weights))

    mean_update_rate = η / 10 # heuristic
    α = mean_update_rate .* p_accept

    update = α .* (proposed.x.v .- [b])
    new_b = 1 / nwalkers(chain_state) * oftype.(b, sum(update .+ [b])) # = (1 - α) * b + α * proposed.x.v

    f_transform_new = MulAdd(Σ_L_new, new_b)

    return f_transform_new, tuner_state_new, chain_state
end
