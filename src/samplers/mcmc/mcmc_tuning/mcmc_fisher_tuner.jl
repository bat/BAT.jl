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
    window::Int
end

# Window of recent draws the low-rank correction is estimated from (the
# diagonal base uses the full foreground-background history). The window
# bounds the estimation memory and fitting cost to O(n_dims * window) and
# the estimable correction rank to 2 * window:
_lowrank_window(max_rank::Integer) = clamp(4 * max(max_rank, 8), 32, 128)

LowRankFisherEstimator(cutoff::Real, max_rank::Integer) =
    LowRankFisherEstimator(cutoff, max_rank, _lowrank_window(max_rank))

_fisher_estimator(at::AbstractAdaptiveTransform) = throw(ArgumentError(
    "FisherTransformTuning requires an affine structure adaptive transform (like TriangularAffineTransform, DiagonalAffineTransform or LowRankAffineTransform), got $(nameof(typeof(at)))"
))
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
[A. Seyboldt, E. L. Carlson and B. Carpenter, "Preconditioning
Hamiltonian Monte Carlo by minimizing Fisher Divergence"
(2026)](https://arxiv.org/abs/2603.18845)).

For an affine transformation `x = A z + μ` with `G = A Aᵀ`, the optimum
satisfies `G Cov(α) G = Cov(x)`, where `α = ∇x log(target)` is the target
score - the affine-invariant geometric mean of the position covariance and
the inverse score covariance. For sufficiently regular targets (vanishing
boundary terms) the score has zero mean and `Cov(α) = E[-∇²log(target)]`,
the average local curvature. For a Gaussian target `Cov(x) = Σ` while
`Cov(α) = Σ⁻¹`, so the optimum is `G = Σ`. The scores come for free: the
z-space gradients that Hamiltonian proposals compute anyway are mapped
back through the current transformation, no additional density or
gradient evaluations are required.

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

    "Regularization added to the diagonal of both covariance estimates,
    relative to their mean variance scale."
    regularization::Float64 = 1e-5
end
export FisherTransformTuning


abstract type _AbstractXGMoments end

# Streaming lag-1 autocovariance of the position observations (diagonal
# only). Interleaved walker observations are handled by striding, so
# lag-1 refers to consecutive draws of the same walker. Provides the
# effective observation count for the drift noise floor:
mutable struct _Lag1Stats
    stride::Int
    n1::Int
    filled::Int
    ptr::Int
    prev::Matrix{Float64}
    cross1::Vector{Float64}
end

_Lag1Stats(n_dims::Integer, stride::Integer) =
    _Lag1Stats(max(stride, 1), 0, 0, 0, zeros(n_dims, max(stride, 1)), zeros(n_dims))

function _lag1_update!(l1::_Lag1Stats, x::AbstractVector{<:Real})
    ptr = mod1(l1.ptr + 1, l1.stride)
    l1.ptr = ptr
    if l1.filled >= l1.stride
        l1.cross1 .+= x .* view(l1.prev, :, ptr)
        l1.n1 += 1
    else
        l1.filled += 1
    end
    l1.prev[:, ptr] .= x
    return l1
end

# Welford accumulator for the means and (co)variances of positions x and
# scores α, dense (M2 matrices) or diagonal (M2 vectors):
mutable struct _XGMoments{M<:Union{Vector{Float64},Matrix{Float64}}} <: _AbstractXGMoments
    n::Int
    mean_x::Vector{Float64}
    mean_g::Vector{Float64}
    M2_x::M
    M2_g::M
    lag1::_Lag1Stats
end

_new_moments(::DenseFisherEstimator, n_dims::Integer, stride::Integer = 1) =
    _XGMoments(0, zeros(n_dims), zeros(n_dims), zeros(n_dims, n_dims), zeros(n_dims, n_dims), _Lag1Stats(n_dims, stride))

_new_moments(::DiagonalFisherEstimator, n_dims::Integer, stride::Integer = 1) =
    _XGMoments(0, zeros(n_dims), zeros(n_dims), zeros(n_dims), zeros(n_dims), _Lag1Stats(n_dims, stride))

# Diagonal Welford moments plus a bounded ring window of recent raw draws
# that the low-rank correction is estimated from - O(n_dims * window)
# memory, no dense moment matrices:
mutable struct _XGWindowMoments <: _AbstractXGMoments
    n::Int
    mean_x::Vector{Float64}
    mean_g::Vector{Float64}
    var_x::Vector{Float64}
    var_g::Vector{Float64}
    X_win::Matrix{Float64}
    G_win::Matrix{Float64}
    win_ptr::Int
    win_count::Int
    lag1::_Lag1Stats
end

_new_moments(est::LowRankFisherEstimator, n_dims::Integer, stride::Integer = 1) =
    _XGWindowMoments(
        0, zeros(n_dims), zeros(n_dims), zeros(n_dims), zeros(n_dims),
        zeros(n_dims, est.window), zeros(n_dims, est.window), 0, 0,
        _Lag1Stats(n_dims, stride)
    )

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
    _lag1_update!(acc.lag1, x)
    return acc
end

function _moments_update!(acc::_XGWindowMoments, x::AbstractVector{<:Real}, g::AbstractVector{<:Real})
    n = (acc.n += 1)
    dx_pre = x .- acc.mean_x
    acc.mean_x .+= dx_pre ./ n
    _m2_update!(acc.var_x, dx_pre, x .- acc.mean_x)
    dg_pre = g .- acc.mean_g
    acc.mean_g .+= dg_pre ./ n
    _m2_update!(acc.var_g, dg_pre, g .- acc.mean_g)
    ptr = mod1(acc.win_ptr + 1, size(acc.X_win, 2))
    acc.win_ptr = ptr
    acc.win_count = min(acc.win_count + 1, size(acc.X_win, 2))
    acc.X_win[:, ptr] .= x
    acc.G_win[:, ptr] .= g
    _lag1_update!(acc.lag1, x)
    return acc
end

_diag_var_raw(acc::_XGMoments{Vector{Float64}}) = acc.M2_x ./ max(acc.n - 1, 1)
_diag_var_raw(acc::_XGMoments{Matrix{Float64}}) = diag(acc.M2_x) ./ max(acc.n - 1, 1)
_diag_var_raw(acc::_XGWindowMoments) = acc.var_x ./ max(acc.n - 1, 1)

# First-order (AR(1)) effective observation count: raw counts overstate
# the information in autocorrelated warmup draws, which would shrink the
# drift noise floor too fast and trigger spurious geometry commits:
function _effective_nobs(acc::_AbstractXGMoments)
    l1 = acc.lag1
    l1.n1 >= 10 || return float(acc.n)
    var_raw = _diag_var_raw(acc)
    c1 = l1.cross1 ./ l1.n1 .- acc.mean_x .^ 2
    ρs = c1 ./ max.(var_raw, floatmin(Float64))
    ρ = clamp(sum(ρs) / length(ρs), 0.0, 0.99)
    return acc.n * (1 - ρ) / (1 + ρ)
end


mutable struct FisherTrafoTunerState{TU<:FisherTransformTuning,E,MO<:_AbstractXGMoments} <: MCMCTransformTunerState
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
    # Pieces (dvec, λ, V) of the last committed low-rank geometry, for the
    # structured drift metric (nothing before the first commit, and for
    # the dense and diagonal estimators):
    committed_lr::Union{Nothing,Tuple{Vector{Float64},Vector{Float64},Matrix{Float64}}}
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
    n_walkers = nwalkers(chain_state)
    memory_length = sched.memory_length > 0 ? sched.memory_length : max(100, 4 * n_dims)
    min_observations = sched.min_observations > 0 ? sched.min_observations : max(20, 2 * n_dims)
    FisherTrafoTunerState(
        tuning, estimator, n_dims, memory_length, min_observations, 0,
        _new_moments(estimator, n_dims, n_walkers), _new_moments(estimator, n_dims, n_walkers),
        nothing
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


# The regularization strength is relative to the mean variance scale of
# each moment matrix, keeping the learned geometry equivariant under a
# global rescaling of the target: positions and scores scale inversely,
# so an absolute floor would swamp whichever side has the smaller scale
# (e.g. the score covariance of a very wide target). Note that the exact
# affine equivariance of the unregularized Fisher equation is only
# preserved under orthogonal and scalar transformations by this scalar
# ridge, not under arbitrary affine maps:
_rel_regularization(γ::Real, C::AbstractMatrix) = γ * max(tr(C) / size(C, 1), floatmin(float(eltype(C))))
_rel_regularization(γ::Real, var_vec::AbstractVector) = γ * max(sum(var_vec) / length(var_vec), floatmin(float(eltype(var_vec))))

# Fisher-optimal affine geometry from the accumulated moments. Returns
# (G, μ) with G the linear geometry (G = A Aᵀ of the optimal transform)
# and μ the Fisher-optimal translation.
function _fisher_geometry(::DenseFisherEstimator, acc::_XGMoments, γ::Real)
    n = acc.n
    C_x_raw = Symmetric(acc.M2_x ./ (n - 1))
    C_g_raw = Symmetric(acc.M2_g ./ (n - 1))
    C_x = Symmetric(C_x_raw + _rel_regularization(γ, C_x_raw) * I)
    C_g = Symmetric(C_g_raw + _rel_regularization(γ, C_g_raw) * I)
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
    var_x_raw = acc.M2_x ./ (n - 1)
    var_g_raw = acc.M2_g ./ (n - 1)
    var_x = var_x_raw .+ _rel_regularization(γ, var_x_raw)
    var_g = var_g_raw .+ _rel_regularization(γ, var_g_raw)
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

# The projected low-rank geometry: a diagonal base from the full moment
# history plus an eigenvalue-thresholded correction fitted in the joint
# thin subspace spanned by the recent standardized position and score
# window (Seyboldt et al. 2026). The window samples span the subspace
# exactly, so the Fisher problem restricted to it is exact for the window
# and the identity geometry is kept outside it - no dense moments and no
# dense solves, everything is O(n_dims * window) plus small-matrix work.
# Returns the geometry, the Fisher-optimal translation and the pieces
# (dvec, λ, V) of the committed representation (for the structured drift):
function _fisher_geometry_lr(est::LowRankFisherEstimator, acc::_XGWindowMoments, γ::Real)
    n = acc.n
    var_x_raw = acc.var_x ./ (n - 1)
    var_g_raw = acc.var_g ./ (n - 1)
    var_x = var_x_raw .+ _rel_regularization(γ, var_x_raw)
    var_g = var_g_raw .+ _rel_regularization(γ, var_g_raw)

    # Diagonal base fit and standardization (scores transform inversely
    # to positions under x̃ = D^{-1/2} x):
    dvec = sqrt.(var_x ./ var_g)
    dsq = sqrt.(dvec)

    m = acc.win_count
    λ, V = if m >= 8
        # The window is centered by its own means: centering by the
        # longer-history means would add a mean-shift outer product to
        # what must be a covariance estimate, turning warmup mean drift
        # into a spurious correction direction. The longer-history means
        # still serve the diagonal base and the translation:
        X_w = view(acc.X_win, :, 1:m)
        G_w = view(acc.G_win, :, 1:m)
        Xc = (X_w .- (sum(X_w, dims = 2) ./ m)) ./ dsq
        Gc = (G_w .- (sum(G_w, dims = 2) ./ m)) .* dsq
        Q = Matrix(qr(hcat(Xc, Gc)).Q)
        Px = Q' * Xc
        Pg = Q' * Gc
        Cx_q_raw = Symmetric(Px * Px' ./ (m - 1))
        Cg_q_raw = Symmetric(Pg * Pg' ./ (m - 1))
        Cx_q = Symmetric(Cx_q_raw + _rel_regularization(γ, Cx_q_raw) * I)
        Cg_q = Symmetric(Cg_q_raw + _rel_regularization(γ, Cg_q_raw) * I)
        Mq = _spd_riccati_solve(Cx_q, Cg_q)
        E = eigen(Symmetric(Matrix(Mq)))
        E.values, Q * E.vectors
    else
        # Not enough window data for a meaningful correction yet:
        Float64[], zeros(length(dvec), 0)
    end

    # G = D^{1/2} (I + V_S (Λ_S - I) V_Sᵀ) D^{1/2} = D + W S Wᵀ:
    W, S, λ_kept, V_kept = _lowrank_correction(dsq, λ, V, est.cutoff, est.max_rank)
    G = woodbury_operator(Diagonal(dvec), W, S)
    μ = acc.mean_x .+ G * acc.mean_g
    return G, μ, (dvec, λ_kept, V_kept)
end

function _fisher_geometry(est::LowRankFisherEstimator, acc::_XGWindowMoments, γ::Real)
    G, μ, _ = _fisher_geometry_lr(est, acc, γ)
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

# Approximate affine-invariant drift between two diagonal-plus-low-rank
# geometries G = D^{1/2} (I + V (Λ - I) Vᵀ) D^{1/2}, without materializing
# dense matrices: the geometry pencil is evaluated exactly within the
# joint correction span (both corrections live there), and by the pure
# diagonal ratios on its complement (both geometries act diagonally
# there); the in-span diagonal-only contribution is subtracted to avoid
# double counting. Exact for pure diagonal changes and for pure
# correction changes; approximate when both interact:
function _lowrank_drift(
    d_o::AbstractVector, λ_o::AbstractVector, V_o::AbstractMatrix,
    d_n::AbstractVector, λ_n::AbstractVector, V_n::AbstractMatrix
)
    tiny = floatmin(Float64)
    r = d_n ./ d_o
    drift2 = sum(x -> abs2(log(max(x, tiny))), r)
    isempty(λ_o) && isempty(λ_n) && return sqrt(drift2)

    # Everything in the D_o-standardized frame, where the new correction
    # directions transport as v -> R v with R = Diagonal(sqrt.(d_n ./ d_o)):
    Rd = sqrt.(r)
    U = Matrix(qr(hcat(Rd .* V_n, V_o)).Q)
    q = size(U, 2)

    UtVo = U' * V_o
    B = Symmetric(Matrix(1.0 * I, q, q) + UtVo * Diagonal(λ_o .- 1) * UtVo')
    RU = Rd .* U
    RUtVn = RU' * V_n
    C = Symmetric(RU' * RU + RUtVn * Diagonal(λ_n .- 1) * RUtVn')
    σ = eigvals(C, B)
    drift2 += sum(x -> abs2(log(max(x, tiny))), σ)

    # In-span diagonal-only contribution, already counted in the ratio term:
    τ = eigvals(Symmetric(Matrix(RU' * RU)))
    drift2 -= sum(x -> abs2(log(max(x, tiny))), τ)

    return sqrt(max(drift2, 0.0))
end

# The drift measurement itself is noisy; its statistical floor scales like
# sqrt(2 n_dims / n_observations), with the effective (autocorrelation-
# corrected) observation count, so the effective commit threshold stays
# above it:
function _effective_commit_threshold(sched::DriftCommitSchedule, n_dims::Integer, n_obs::Real)
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
        tuner_state.acc_b = _new_moments(tuner_state.estimator, tuner_state.n_dims, tuner_state.acc_b.lag1.stride)
    end

    sched = tuner_state.tuning.schedule
    acc = tuner_state.acc_a
    if tuner_state.nsteps % sched.check_interval == 0 && acc.n >= tuner_state.min_observations
        est = tuner_state.estimator
        local G, μ, drift
        lr_pieces = nothing
        if est isa LowRankFisherEstimator
            G, μ, lr_pieces = _fisher_geometry_lr(est, acc, tuner_state.tuning.regularization)
            # The installed initial geometry's pieces are unknown (the
            # transform factor is an opaque operator), and a dense drift
            # comparison against it would defeat the estimator's
            # high-dimensional scaling. So the first statistically
            # eligible estimate always commits, becoming the baseline;
            # from then on the structured drift metric decides, and
            # nothing on the low-rank path ever materializes a dense
            # geometry:
            drift = isnothing(tuner_state.committed_lr) ? oftype(sched.commit_threshold, Inf) :
                _lowrank_drift(tuner_state.committed_lr..., lr_pieces...)
        else
            G, μ = _fisher_geometry(est, acc, tuner_state.tuning.regularization)
            drift = _transform_drift(A, G)
        end
        if drift > _effective_commit_threshold(sched, tuner_state.n_dims, _effective_nobs(acc))
            tuner_state.committed_lr = lr_pieces
            A_new = _fisher_A(est, G)
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
