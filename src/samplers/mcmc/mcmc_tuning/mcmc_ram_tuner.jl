# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct RAMTuning <: MCMCTransformTuning

Tunes MCMC spaces transformations based on
[M. Vihola, "Robust adaptive Metropolis algorithm with coerced
acceptance rate" (2012)](https://doi.org/10.1007/s11222-011-9269-5).

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

# RAM drifts the transformation in small per-step updates, step-size
# adaptation tracks it instead of restarting:
transform_change_restarts_stepsize(::RAMTrafoTunerState) = false


function create_trafo_tuner_state(
    tuning::RAMTuning,
    chain_state::MCMCChainState,
    n_steps_hint::Integer
)
    f = chain_state.f_transform
    (f isa MulAdd && f.A isa LowerTriangular) || throw(ArgumentError(
        "RAMTuning requires a TriangularAffineTransform adaptive space transformation"
    ))
    RAMTrafoTunerState(tuning, 0)
end

function mcmc_trafo_tuning_init!!(
    tuner_state::RAMTrafoTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false) # TODO ?
    tuner_state.nsteps = 0
    return nothing
end

# Computes the lower Cholesky factor of `L * (I + sum_i w_i * u_i * u_iᵀ) * Lᵀ`.
# Single-walker chains (a single update vector) use a rank-one Cholesky
# up/downdate in O(n²) where possible, following the original RAM
# formulation (Vihola 2012). Multi-walker chains use a full modified
# Cholesky decomposition instead: BLAS-3 kernels beat sequential rank-one
# sweeps already at moderate walker counts, and aggregated mixed-sign
# walker updates can make the exact update indefinite, so the modified
# decomposition is required as a positive-definiteness floor there anyway.
function _rank_k_cholesky_update(L::LowerTriangular{<:Real}, u::AbstractVector{<:AbstractVector{<:Real}}, w::AbstractVector{<:Real})
    if length(u) == 1 && length(w) == 1
        return _rank_one_cholesky_update(L, u[1], w[1])
    else
        return _modified_cholesky_update(L, u, w)
    end
end

# Rank-one Cholesky downdate of the lower-triangular factor stored in `A`,
# in place, destroying `v` (LINPACK dchdd-style recurrence, like
# `LinearAlgebra.lowrankdowndate!`). Returns `true` on success and `false`
# if the downdated matrix is not numerically positive definite, instead of
# throwing or silently producing an invalid factor like the stdlib
# implementation can (its strict `s² > 1` test lets `s² == 1` corrupt the
# factor via a zero pivot, and NaN pass through). The single check
# `c² > 0` covers indefiniteness and non-finite poisoning at the point
# where they arise; the contents of `A` and `v` are undefined after
# failure. The strict upper triangle of `A` is never referenced:
function _lowrankdowndate!(A::AbstractMatrix{<:Real}, v::AbstractVector{<:Real})
    n = length(v)
    for i in 1:n
        Aii = A[i,i]
        s = v[i] / Aii
        c2 = 1 - abs2(s)
        c2 > 0 || return false
        c = sqrt(c2)
        A[i,i] = c * Aii
        for j in (i + 1):n
            vj = v[j]
            Aji = (A[j,i] - s * vj) / c
            A[j,i] = Aji
            v[j] = -s * Aji + c * vj
        end
    end
    return true
end

# A rank-one downdate keeps the factor positive definite iff ‖L⁻¹v‖² < 1.
# With `v = sqrt(|w1|) * L * u1` the statistic `‖L⁻¹v‖² = |w1| * ‖u1‖²` is
# exact and independent of the conditioning of `L`; the acceptance
# criterion adds a conservative numerical margin below the boundary. It
# cannot rule out failure of the numerical downdate for very
# ill-conditioned `L`, though, since the downdate recurrence operates on
# the rounded `v`, so the downdate reports failure instead of relying on
# a-priori feasibility. Rejected and failed downdates and non-finite
# values (from numerically singular factors, they fail either comparison)
# take the modified-decomposition floor instead, which is the desired
# regularization for them anyway: rank-one downdates would degenerate
# under long streaks of low acceptance.
function _rank_one_cholesky_update(L::LowerTriangular{<:Real}, u1::AbstractVector{<:Real}, w1::Real)
    v = sqrt(abs(w1)) .* (L * u1)
    margin = sqrt(eps(one(eltype(v))))
    feasible = all(isfinite, v) && (w1 >= 0 || abs(w1) * sum(abs2, u1) < 1 - margin)
    if feasible
        # Mutable dense copy that preserves the storage type of L, unlike
        # Matrix(L) would; the parent must be a mutable dense matrix. Its
        # strict upper triangle is never referenced:
        A = copy(parent(L))
        if w1 >= 0
            lowrankupdate!(Cholesky(LowerTriangular(A)), v)
            return LowerTriangular(A)
        elseif _lowrankdowndate!(A, v)
            return LowerTriangular(A)
        end
    end
    return _modified_cholesky_update(L, (u1,), (w1,))
end

# Modified Cholesky decomposition, used for multi-walker updates and as
# the positive-definiteness floor when a rank-one up-/downdate is not
# applicable:
function _modified_cholesky_update(L::LowerTriangular{<:Real}, u, w)
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
    step_info::MCMCStepInfo
)
    if any(current.x.v .== proposed.x.v)
        return f_transform, tuner_state, chain_state
    end

    walker_order = step_info.walker_order
    p_accept = step_info.p_accept[walker_order]

    gamma = tuner_state.tuning.gamma
    target_acceptance = get_target_acceptance_ratio(proposal)
    b = f_transform.b
    n_dims = length(b)

    tuner_state_new = @set tuner_state.nsteps = tuner_state.nsteps + 1

    η = min(1, n_dims * tuner_state.nsteps^(-gamma))

    Σ_L = f_transform.A

    u = proposed.z.v[walker_order] .- current.z.v[walker_order]
    weights = (p_accept .- target_acceptance) ./ norm.(u).^2
    Σ_L_new = oftype(Σ_L, _rank_k_cholesky_update(Σ_L, u, η .* weights))

    mean_update_rate = η / 10 # heuristic
    α = mean_update_rate .* p_accept

    update = α .* (proposed.x.v[walker_order] .- [b])
    new_b = 1 / nwalkers(chain_state) * oftype.(b, sum(update .+ [b])) # = (1 - α) * b + α * proposed.x.v

    f_transform_new = MulAdd(Σ_L_new, new_b)

    return f_transform_new, tuner_state_new, chain_state
end


# The transform-tuning default depends on the proposal as well: the tuning
# rule must match the statistics the proposal generates. RAM drifts a
# dense triangular geometry from the accepted/rejected steps alone, so it
# is the default for proposals that provide no gradients (see
# FisherTransformTuning for those that do):
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::MCMCProposal, ::TriangularAffineTransform) = RAMTuning()
