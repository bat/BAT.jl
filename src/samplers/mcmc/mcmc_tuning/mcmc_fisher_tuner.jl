# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# The geometry-estimation structure follows the declared adaptive
# transform (see _fisher_estimator), users select it by choosing a
# TriangularAffineTransform, DiagonalAffineTransform or
# LowRankAffineTransform:

struct DenseFisherEstimator end

struct DiagonalFisherEstimator end

struct LowRankFisherEstimator
    cutoff::Float64
    max_rank::Int
end

_fisher_estimator(::TriangularAffineTransform) = DenseFisherEstimator()
_fisher_estimator(::DiagonalAffineTransform) = DiagonalFisherEstimator()
function _fisher_estimator(at::LowRankAffineTransform)
    @argcheck at.cutoff > 1
    @argcheck at.max_rank >= 0
    LowRankFisherEstimator(at.cutoff, at.max_rank)
end

# Fallback when the transform declaration is not available (direct
# low-level use): infer the structure from the installed transform:
_fisher_estimator_from_A(::LowerTriangular) = DenseFisherEstimator()
_fisher_estimator_from_A(::Diagonal) = DiagonalFisherEstimator()
_fisher_estimator_from_A(::Any) = DenseFisherEstimator()


"""
    struct DriftCommitSchedule

Transform-installation policy of [`FisherTransformTuning`](@ref): geometry
statistics accumulate continuously, but a new transformation is only
committed when the estimated geometry has drifted away from the installed
one by more than a threshold in the affine-invariant SPD metric (plus a
statistical noise floor). Early in warmup, noisy estimates drift fast and
commits are frequent; as the estimate converges, commits cease on their
own - there are no scheduled adaptation windows.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct DriftCommitSchedule
    "Commit threshold in the affine-invariant SPD metric (a statistical
    noise floor is added automatically)."
    commit_threshold::Float64 = 0.3

    "Steps between drift evaluations."
    check_interval::Int = 10

    "Steps per foreground-background estimator memory block, `0` selects
    an automatic, dimension-derived length."
    memory_length::Int = 0

    "Minimum number of accumulated observations before the first commit,
    `0` selects an automatic, dimension-derived count."
    min_observations::Int = 0
end
export DriftCommitSchedule


"""
    struct FisherTransformTuning <: MCMCTransformTuning

Tunes MCMC space transformations for gradient-based proposals (currently
[`HamiltonianMC`](@ref)) by minimizing the empirical Fisher divergence of
the transformed target to a standard normal distribution (following
Seyboldt, Carlson & Carpenter, "Preconditioning Hamiltonian Monte Carlo by
minimizing Fisher Divergence", 2026).

For an affine transformation `x = A z + μ` with `G = A Aᵀ`, the optimum
satisfies `G Cov(α) G = Cov(x)`, where `α = ∇x log(target)` is the target
score - the affine-invariant geometric mean of the position covariance and
the inverse score covariance. Both statistics coincide with the target
covariance for Gaussian targets; elsewhere the score side contributes the
average local curvature (`Cov(α) = E[-∇²log(target)]`). The scores come
for free: the z-space gradients that Hamiltonian proposals compute anyway
are mapped back through the current transformation, no additional density
or gradient evaluations are required.

Positions and scores are accumulated in the fixed pre-adaptive space with
foreground-background memory (early, transient-contaminated draws are
periodically forgotten). Transform updates follow the `schedule`; each
committed transform triggers a fresh step-size search and a dual-averaging
restart in the step-size adaptor (see [`BAT.StepSizeAdaptor`](@ref)).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct FisherTransformTuning{S} <: MCMCTransformTuning
    "Transform-installation policy."
    schedule::S = DriftCommitSchedule()

    "Regularization added to the diagonal of both covariance estimates."
    regularization::Float64 = 1e-5
end
export FisherTransformTuning


# Welford accumulator for the means and (co)variances of positions x and
# scores α, dense (M2 matrices) or diagonal (M2 vectors):
mutable struct _XGMoments{M<:Union{Vector{Float64},Matrix{Float64}}}
    n::Int
    mean_x::Vector{Float64}
    mean_g::Vector{Float64}
    M2_x::M
    M2_g::M
end

_new_moments(::DenseFisherEstimator, n_dims::Integer) =
    _XGMoments(0, zeros(n_dims), zeros(n_dims), zeros(n_dims, n_dims), zeros(n_dims, n_dims))

_new_moments(::DiagonalFisherEstimator, n_dims::Integer) =
    _XGMoments(0, zeros(n_dims), zeros(n_dims), zeros(n_dims), zeros(n_dims))

_new_moments(::LowRankFisherEstimator, n_dims::Integer) = _new_moments(DenseFisherEstimator(), n_dims)

_m2_update!(M2::Matrix{Float64}, d_pre, d_post) = (M2 .+= d_pre .* d_post')
_m2_update!(M2::Vector{Float64}, d_pre, d_post) = (M2 .+= d_pre .* d_post)

function _moments_update!(acc::_XGMoments, x::AbstractVector{<:Real}, g::AbstractVector{<:Real})
    n = (acc.n += 1)
    dx_pre = x .- acc.mean_x
    acc.mean_x .+= dx_pre ./ n
    _m2_update!(acc.M2_x, dx_pre, x .- acc.mean_x)
    dg_pre = g .- acc.mean_g
    acc.mean_g .+= dg_pre ./ n
    _m2_update!(acc.M2_g, dg_pre, g .- acc.mean_g)
    return acc
end


mutable struct FisherTrafoTunerState{TU<:FisherTransformTuning,E,MO<:_XGMoments} <: MCMCTransformTunerState
    tuning::TU
    estimator::E
    n_dims::Int
    memory_length::Int
    min_observations::Int
    nsteps::Int
    # Foreground-background memory: both accumulators receive every
    # observation; every memory_length steps the older one (which
    # estimates are computed from) is replaced by the younger one and the
    # younger one starts fresh, so estimates always cover between one and
    # two memory blocks of recent history and forget the transient-
    # contaminated early draws:
    acc_a::MO
    acc_b::MO
end

# Whether a proposal provides z-space log-density gradients in its
# MCMCStepInfo (see mcmc_step_provides_grads).
function create_trafo_tuner_state(
    tuning::FisherTransformTuning,
    chain_state::MCMCChainState,
    n_steps_hint::Integer,
    adaptive_transform::AbstractAdaptiveTransform
)
    _create_fisher_tuner_state(tuning, chain_state, _fisher_estimator(adaptive_transform))
end

function create_trafo_tuner_state(
    tuning::FisherTransformTuning,
    chain_state::MCMCChainState,
    n_steps_hint::Integer
)
    _create_fisher_tuner_state(tuning, chain_state, _fisher_estimator_from_A(chain_state.f_transform.A))
end

function _create_fisher_tuner_state(tuning::FisherTransformTuning, chain_state::MCMCChainState, estimator)
    sched = tuning.schedule
    @argcheck sched.check_interval >= 1
    @argcheck sched.commit_threshold >= 0
    @argcheck sched.memory_length >= 0
    @argcheck sched.min_observations >= 0
    @argcheck tuning.regularization > 0

    proposal = get_active_proposal(chain_state.proposal)
    mcmc_step_provides_grads(proposal) || throw(ArgumentError(
        "FisherTransformTuning requires an MCMC proposal that provides log-density gradients (like HamiltonianMC or MALAProposal), got $(nameof(typeof(proposal)))"
    ))
    chain_state.f_transform isa MulAdd || throw(ArgumentError(
        "FisherTransformTuning requires an affine adaptive space transformation (like TriangularAffineTransform)"
    ))
    n_dims = length(first(chain_state.current.x.v))
    memory_length = sched.memory_length > 0 ? sched.memory_length : max(100, 4 * n_dims)
    min_observations = sched.min_observations > 0 ? sched.min_observations : max(20, 2 * n_dims)
    FisherTrafoTunerState(
        tuning, estimator, n_dims, memory_length, min_observations, 0,
        _new_moments(estimator, n_dims), _new_moments(estimator, n_dims)
    )
end

function mcmc_trafo_tuning_init!!(
    tuner_state::FisherTrafoTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer
)
    chain_state.info = MCMCChainStateInfo(chain_state.info, tuned = false)
    return nothing
end

# No mcmc_trafo_tuning_reinit!! specialization: estimation continues
# seamlessly across burn-in cycles (the generic fallback is a no-op).


# Fisher-optimal affine geometry from the accumulated moments. Returns
# (G, μ) with G the linear geometry (G = A Aᵀ of the optimal transform)
# and μ the Fisher-optimal translation.
function _fisher_geometry(::DenseFisherEstimator, acc::_XGMoments, γ::Real)
    n = acc.n
    C_x = Symmetric(acc.M2_x ./ (n - 1) + γ * I)
    C_g = Symmetric(acc.M2_g ./ (n - 1) + γ * I)
    G = _spd_riccati_solve(C_x, C_g)
    # The Fisher-optimal translation is the score-corrected mean
    # μ = x̄ + G ᾱ (it reduces to x̄ at stationarity, where E[α] = 0).
    # Note: the 2026 Fisher-HMC paper's (v1) appendix C theorem statement
    # writes the mass-matrix form of this with M in place of M⁻¹ = G, in
    # contradiction to its own derivation and its diagonal theorem:
    μ = acc.mean_x .+ G * acc.mean_g
    return G, μ
end

function _fisher_geometry(::DiagonalFisherEstimator, acc::_XGMoments, γ::Real)
    n = acc.n
    var_x = acc.M2_x ./ (n - 1) .+ γ
    var_g = acc.M2_g ./ (n - 1) .+ γ
    g = sqrt.(var_x ./ var_g)
    μ = acc.mean_x .+ g .* acc.mean_g
    return Diagonal(g), μ
end

# Unique SPD solution G of the Riccati equation G C_g G = C_x, i.e. the
# affine-invariant geometric mean of C_x and C_g⁻¹:
function _spd_riccati_solve(C_x::Symmetric, C_g::Symmetric)
    E = eigen(C_g)
    S_sqrt = E.vectors * Diagonal(sqrt.(E.values)) * E.vectors'
    S_isqrt = E.vectors * Diagonal(inv.(sqrt.(E.values))) * E.vectors'
    F = eigen(Symmetric(S_sqrt * C_x * S_sqrt))
    M_sqrt = F.vectors * Diagonal(sqrt.(max.(F.values, 0))) * F.vectors'
    return Symmetric(S_isqrt * M_sqrt * S_isqrt)
end

# The low-rank geometry: a diagonal base plus an eigenvalue-thresholded
# correction along the directions where the diagonal alone is
# insufficient. Fitting only these few directions regularizes the
# estimate compared to a full dense geometry:
function _fisher_geometry(est::LowRankFisherEstimator, acc::_XGMoments, γ::Real)
    n = acc.n
    C_x = Symmetric(acc.M2_x ./ (n - 1) + γ * I)
    C_g = Symmetric(acc.M2_g ./ (n - 1) + γ * I)

    # Diagonal base fit and standardization (scores transform inversely
    # to positions under x̃ = D^{-1/2} x):
    dvec = sqrt.(diag(C_x) ./ diag(C_g))
    dsq = sqrt.(dvec)
    Ct_x = Symmetric(C_x ./ (dsq .* dsq'))
    Ct_g = Symmetric(C_g .* (dsq .* dsq'))

    Gt = _spd_riccati_solve(Ct_x, Ct_g)
    E = eigen(Symmetric(Matrix(Gt)))
    λ, V = E.values, E.vectors
    keep = findall(l -> l > est.cutoff || l < inv(est.cutoff), λ)
    if est.max_rank > 0 && length(keep) > est.max_rank
        keep = keep[sortperm(abs.(log.(λ[keep])), rev = true)[1:est.max_rank]]
    end

    # G = D^{1/2} (I + V_S (Λ_S - I) V_Sᵀ) D^{1/2} = D + W S Wᵀ:
    W = (dsq .* V[:, keep]) .* sqrt.(abs.(λ[keep] .- 1))'
    S = Symmetric(Matrix(Diagonal(sign.(λ[keep] .- 1))))
    G = woodbury_operator(Diagonal(dvec), W, S)
    μ = acc.mean_x .+ G * acc.mean_g
    return G, μ
end

_dense_spd(G::Symmetric) = G
_dense_spd(G::Diagonal) = Symmetric(Matrix(G))
_dense_spd(G) = Symmetric(Matrix(G))

# The matrix part of the transform to commit, in the structure the
# estimator maintains:
_fisher_A(::DenseFisherEstimator, G) = cholesky(Positive, Matrix(_dense_spd(G))).L
_fisher_A(::DiagonalFisherEstimator, G::Diagonal) = Diagonal(sqrt.(G.diag))
_fisher_A(::LowRankFisherEstimator, G) = rowgram_factor(G)

# Affine-invariant SPD distance between the installed geometry
# G_inst = A Aᵀ and the estimated geometry G. The spectrum of A⁻¹ G A⁻ᵀ
# equals the spectrum of G_inst⁻¹ G, and A⁻¹ G A⁻ᵀ = A \ ((A \ G)ᵀ)
# for symmetric G, so only (adjoint-free) left solves are needed:
function _transform_drift(A, G)
    Gd = Matrix(_dense_spd(G))
    B = Symmetric(Matrix(A \ Matrix((A \ Gd)')))
    λ = eigvals(B)
    tiny = floatmin(float(eltype(λ)))
    return sqrt(sum(x -> abs2(log(max(x, tiny))), λ))
end

# The drift measurement itself is noisy; its statistical floor scales like
# sqrt(2 n_dims / n_observations), so the effective commit threshold stays
# above it:
function _effective_commit_threshold(sched::DriftCommitSchedule, n_dims::Integer, n_obs::Integer)
    return sched.commit_threshold + 3 * sqrt(2 * n_dims / max(n_obs, 1))
end

function mcmc_tune_trafo_post_step!!(
    f_transform::Function,
    tuner_state::FisherTrafoTunerState,
    chain_state::MCMCChainState,
    proposal::MCMCProposalState,
    current::NamedTuple{<:Any,<:Tuple{Vararg{DensitySampleVector}}},
    proposed::NamedTuple{<:Any,<:Tuple{Vararg{DensitySampleVector}}},
    step_info::MCMCStepInfo
)
    z_grads = step_info.z_grads
    isnothing(z_grads) && return f_transform, tuner_state, chain_state

    A = f_transform.A
    xs_prop = proposed.x.v
    xs_curr = current.x.v
    accepted = chain_state.accepted
    for i in eachindex(xs_prop, z_grads)
        # Score transport into the fixed pre-adaptive space: for x = A z + μ
        # the pulled-back gradient is β = Aᵀ α, so α = A⁻ᵀ β:
        α = A' \ z_grads[i]
        # The gradients refer to the selected (post-accept/reject) states,
        # so the positions must too - on rejection the selected state is
        # the current one, not the rejected proposal. A repeated state is
        # a valid draw as well; accumulate in both memory blocks:
        x_i = accepted[i] ? xs_prop[i] : xs_curr[i]
        _moments_update!(tuner_state.acc_a, x_i, α)
        _moments_update!(tuner_state.acc_b, x_i, α)
    end

    tuner_state.nsteps += 1
    if tuner_state.nsteps % tuner_state.memory_length == 0
        tuner_state.acc_a = tuner_state.acc_b
        tuner_state.acc_b = _new_moments(tuner_state.estimator, tuner_state.n_dims)
    end

    sched = tuner_state.tuning.schedule
    acc = tuner_state.acc_a
    if tuner_state.nsteps % sched.check_interval == 0 && acc.n >= tuner_state.min_observations
        G, μ = _fisher_geometry(tuner_state.estimator, acc, tuner_state.tuning.regularization)
        drift = _transform_drift(A, G)
        if drift > _effective_commit_threshold(sched, tuner_state.n_dims, acc.n)
            A_new = _fisher_A(tuner_state.estimator, G)
            b_new = oftype(f_transform.b, μ)
            return MulAdd(A_new, b_new), tuner_state, chain_state
        end
    end

    return f_transform, tuner_state, chain_state
end


# Gradient-based proposals default to Fisher-divergence transform tuning
# for all affine transform structures:
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::Union{HamiltonianMC,MALAProposal}, ::TriangularAffineTransform) = FisherTransformTuning()
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::Union{HamiltonianMC,MALAProposal}, ::Union{DiagonalAffineTransform,LowRankAffineTransform}) = FisherTransformTuning()

function bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, proposal::MCMCProposal, ::Union{DiagonalAffineTransform,LowRankAffineTransform})
    throw(ArgumentError("Diagonal and low-rank affine transform tuning currently requires a gradient-based MCMC proposal (like HamiltonianMC or MALAProposal), not $(nameof(typeof(proposal)))"))
end
