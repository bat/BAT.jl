# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# The geometry-estimation structure follows the declared adaptive
# transform (see _fisher_estimator), users select it by choosing a
# subtype of BAT.AbstractAffineTransform:

struct DenseFisherEstimator end

struct DiagonalFisherEstimator end

struct LowRankFisherEstimator
    cutoff::Float64
    max_rank::Int
end

LowRankFisherEstimator(cutoff::Real, max_rank::Integer) =
    LowRankFisherEstimator(Float64(cutoff), Int(max_rank))

const _LR_DYNAMIC_MAX_DIMS = 32

@enum _LowRankPhase::UInt8 _LRWaiting _LRFit _LRGuard _LRValidate _LRFrozen

mutable struct _XGFitBlock{T<:AbstractFloat}
    X::Matrix{T}
    G::Matrix{T}
    nsteps::Int
end

struct _LowRankCandidate{T<:AbstractFloat}
    lambda::Vector{T}
    vectors::Matrix{T}
    W::Matrix{T}
    S::Symmetric{T,Matrix{T}}
end

mutable struct _LowRankCampaign{T<:AbstractFloat}
    phase::_LowRankPhase
    attempted::Bool
    admitted::Bool
    cycle_step::Int
    fit_start::Int
    fit_steps::Int
    guard_steps::Int
    validation_steps::Int
    final_steps::Int
    fit::_XGFitBlock{T}
    validation_loss::Matrix{T}
    validation_offdiag_loss::Matrix{T}
    baseline_dvec::Union{Nothing,Vector{T}}
    baseline_mu::Union{Nothing,Vector{T}}
    candidate::Union{Nothing,_LowRankCandidate{T}}
end

function _LowRankCampaign(
    ::Type{T},
    n_dims::Integer,
    max_nsteps::Integer,
    n_walkers::Integer,
    is_mala::Bool = false,
) where {T<:AbstractFloat}
    n_dims <= _LR_DYNAMIC_MAX_DIMS || return nothing
    n_dims > 0 || return nothing
    n_walkers > 0 || return nothing

    fit_steps = n_dims >= 16 ? max(2 * n_dims, 64) : 2 * n_dims
    guard_steps = is_mala ? max(n_dims, 64) : n_dims
    validation_steps = is_mala ? max(16 * n_dims, 512) : max(8 * n_dims, 256)
    final_steps = max(100, ceil(Int, 0.15 * max_nsteps))
    fit_start = floor(Int, (is_mala ? 0.20 : 0.30) * max_nsteps) + 1
    decision_end = fit_start + fit_steps + guard_steps + validation_steps - 1
    deadline = floor(Int, 0.85 * max_nsteps)

    decision_end <= deadline || return nothing
    max_nsteps - decision_end >= final_steps || return nothing

    return _LowRankCampaign{T}(
        _LRWaiting, false, false, 0, fit_start, fit_steps, guard_steps,
        validation_steps, final_steps,
        _XGFitBlock(
            zeros(T, n_dims, fit_steps * n_walkers),
            zeros(T, n_dims, fit_steps * n_walkers),
            0
        ),
        zeros(T, n_walkers, validation_steps),
        zeros(T, n_walkers, validation_steps),
        nothing, nothing, nothing
    )
end

_LowRankCampaign(
    n_dims::Integer,
    max_nsteps::Integer,
    n_walkers::Integer,
    is_mala::Bool = false,
) = _LowRankCampaign(Float64, n_dims, max_nsteps, n_walkers, is_mala)

function _diagonal_lowrank_campaign(
    ::Type{T},
    n_dims::Integer,
    max_nsteps::Integer,
    n_walkers::Integer,
) where {T<:AbstractFloat}
    freeze_step = floor(Int, 0.85 * max_nsteps)
    return _LowRankCampaign{T}(
        _LRWaiting, false, false, 0, freeze_step + 1, 0, 0, 0,
        max_nsteps - freeze_step,
        _XGFitBlock(zeros(T, n_dims, 0), zeros(T, n_dims, 0), 0),
        zeros(T, n_walkers, 0),
        zeros(T, n_walkers, 0),
        nothing, nothing, nothing
    )
end

function _advance_lowrank_campaign!(campaign::_LowRankCampaign, cycle_step::Integer)
    campaign.attempted && return campaign
    campaign.cycle_step = cycle_step
    fit_end = campaign.fit_start + campaign.fit_steps - 1
    guard_end = fit_end + campaign.guard_steps
    validation_end = guard_end + campaign.validation_steps

    if cycle_step < campaign.fit_start
        campaign.phase = _LRWaiting
    elseif cycle_step <= fit_end
        campaign.phase = _LRFit
    elseif cycle_step <= guard_end
        campaign.phase = _LRGuard
    elseif cycle_step <= validation_end
        campaign.phase = _LRValidate
    else
        campaign.phase = _LRFrozen
        campaign.attempted = true
    end
    return campaign
end

_fisher_estimator(at::AbstractAdaptiveTransform) = throw(ArgumentError(
    "FisherTransformTuning requires an affine adaptive space transformation (a subtype of BAT.AbstractAffineTransform, like TriangularAffineTransform), got $(nameof(typeof(at)))"
))
_fisher_estimator(at::AbstractAffineTransform) = throw(ArgumentError(
    "$(nameof(typeof(at))) does not implement BAT._fisher_estimator, so FisherTransformTuning can't determine its geometry-estimation structure"
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

Transform-installation policy of [`FisherTransformTuning`](@ref).

Geometry statistics accumulate continuously, but a new transformation is
committed only when the estimated geometry has drifted far enough from the
installed one: their distance in the affine-invariant SPD metric must
exceed `commit_threshold` plus a statistical noise floor.

Early in warmup, noisy estimates drift fast and commits are frequent; as
the estimate converges, commits cease on their own - there are no
scheduled adaptation windows.

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

Fisher moment, fit, and validation state uses the chain's floating-point
type.

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
mutable struct _Lag1Stats{T<:AbstractFloat}
    stride::Int
    n1::Int
    filled::Int
    ptr::Int
    prev::Matrix{T}
    cross1::Vector{T}
end

function _Lag1Stats(::Type{T}, n_dims::Integer, stride::Integer) where {T<:AbstractFloat}
    stride = max(stride, 1)
    _Lag1Stats(stride, 0, 0, 0, zeros(T, n_dims, stride), zeros(T, n_dims))
end

_Lag1Stats(n_dims::Integer, stride::Integer) = _Lag1Stats(Float64, n_dims, stride)

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
mutable struct _XGMoments{T<:AbstractFloat,M<:Union{Vector{T},Matrix{T}}} <: _AbstractXGMoments
    n::Int
    mean_x::Vector{T}
    mean_g::Vector{T}
    M2_x::M
    M2_g::M
    lag1::_Lag1Stats{T}
end

_new_moments(
    ::DenseFisherEstimator,
    ::Type{T},
    n_dims::Integer,
    stride::Integer = 1,
) where {T<:AbstractFloat} = _XGMoments(0, zeros(T, n_dims), zeros(T, n_dims),
    zeros(T, n_dims, n_dims), zeros(T, n_dims, n_dims), _Lag1Stats(T, n_dims, stride))

_new_moments(
    ::DiagonalFisherEstimator,
    ::Type{T},
    n_dims::Integer,
    stride::Integer = 1,
) where {T<:AbstractFloat} = _XGMoments(0, zeros(T, n_dims), zeros(T, n_dims),
    zeros(T, n_dims), zeros(T, n_dims), _Lag1Stats(T, n_dims, stride))

_new_moments(
    est::LowRankFisherEstimator,
    T::Type{<:AbstractFloat},
    n_dims::Integer,
    stride::Integer = 1,
) = _new_moments(DiagonalFisherEstimator(), T, n_dims, stride)

_new_moments(est, prototype::AbstractVector{P}, stride::Integer = 1) where {P<:Real} =
    _new_moments(est, float(P), length(prototype), stride)

_new_moments(est, n_dims::Integer, stride::Integer = 1) =
    _new_moments(est, Float64, n_dims, stride)

_m2_update!(M2::AbstractMatrix, d_pre, d_post) = (M2 .+= d_pre .* d_post')
_m2_update!(M2::AbstractVector, d_pre, d_post) = (M2 .+= d_pre .* d_post)

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

_diag_var_raw(acc::_XGMoments{T,M}) where {T,M<:AbstractVector} = acc.M2_x ./ max(acc.n - 1, 1)
_diag_var_raw(acc::_XGMoments{T,M}) where {T,M<:AbstractMatrix} = diag(acc.M2_x) ./ max(acc.n - 1, 1)

# First-order (AR(1)) effective observation count: raw counts overstate
# the information in autocorrelated warmup draws, which would shrink the
# drift noise floor too fast and trigger spurious geometry commits:
function _effective_nobs(acc::_AbstractXGMoments)
    l1 = acc.lag1
    T = eltype(acc.mean_x)
    l1.n1 >= 10 || return T(acc.n)
    var_raw = _diag_var_raw(acc)
    c1 = l1.cross1 ./ l1.n1 .- acc.mean_x .^ 2
    ρs = c1 ./ max.(var_raw, floatmin(T))
    ρ = clamp(sum(ρs) / length(ρs), zero(T), T(99) / 100)
    return acc.n * (1 - ρ) / (1 + ρ)
end


mutable struct FisherTrafoTunerState{TU<:FisherTransformTuning,E,MO<:_AbstractXGMoments,T<:AbstractFloat,CS} <: MCMCTransformTunerState
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
    # Last committed diagonal base for the low-rank path. The correction
    # campaign has one decision and therefore needs no rolling baseline.
    committed_diag::Union{Nothing,Vector{T}}
    campaign::CS
end

function transform_tuning_pauses_proposal(tuner::FisherTrafoTunerState)
    campaign = tuner.campaign
    return !isnothing(campaign) && campaign.phase == _LRValidate
end

function mcmc_proposal_transform_committed!!(
    proposal::MALAProposalState,
    tuner::MALAStepSizeTunerState,
    chain_state::MCMCChainState,
    trafo_tuner::FisherTrafoTunerState,
)
    _reset_mala_stepsize_tuner!(tuner, chain_state)
    campaign = trafo_tuner.campaign
    if !isnothing(campaign) && campaign.phase in (_LRFit, _LRFrozen)
        tuner.min_run_nobs = min(40, campaign.final_steps) * nwalkers(chain_state)
    end
    return proposal, tuner, chain_state
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
    prototype = first(chain_state.current.x.v)
    n_dims = length(prototype)
    n_walkers = nwalkers(chain_state)
    memory_length = sched.memory_length > 0 ? sched.memory_length : max(100, 4 * n_dims)
    min_observations = sched.min_observations > 0 ? sched.min_observations : max(20, 2 * n_dims)
    acc_a = _new_moments(estimator, prototype, n_walkers)
    acc_b = _new_moments(estimator, prototype, n_walkers)
    T = eltype(acc_a.mean_x)
    campaign_type = estimator isa LowRankFisherEstimator ?
        Union{Nothing,_LowRankCampaign{T}} : Nothing
    FisherTrafoTunerState{typeof(tuning),typeof(estimator),typeof(acc_a),T,campaign_type}(
        tuning, estimator, n_dims, memory_length, min_observations, 0,
        acc_a, acc_b, nothing, nothing
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

function mcmc_trafo_tuning_reinit!!(
    tuner_state::FisherTrafoTunerState,
    chain_state::MCMCChainState,
    max_nsteps::Integer,
)
    est = tuner_state.estimator
    if est isa LowRankFisherEstimator
        dynamic_eligible =
            est.cutoff >= 1.5 && tuner_state.n_dims <= _LR_DYNAMIC_MAX_DIMS
        campaign = tuner_state.campaign
        retryable = isnothing(campaign) ||
            (dynamic_eligible && campaign.fit_steps == 0)
        retryable || return nothing

        n_walkers = nwalkers(chain_state)
        T = eltype(tuner_state.acc_a.mean_x)
        campaign = if !dynamic_eligible
            _diagonal_lowrank_campaign(T, tuner_state.n_dims, max_nsteps, n_walkers)
        else
            candidate = _LowRankCampaign(
                T,
                tuner_state.n_dims,
                max_nsteps,
                n_walkers,
                get_active_proposal(chain_state.proposal) isa MALAProposalState,
            )
            isnothing(candidate) && !isnothing(campaign) && return nothing
            something(
                candidate,
                _diagonal_lowrank_campaign(
                    T, tuner_state.n_dims, max_nsteps, n_walkers,
                ),
            )
        end
        tuner_state.campaign = campaign
    end
    return nothing
end

# The regularization strength is relative to the mean variance scale of
# each moment matrix, keeping the learned geometry equivariant under a
# global rescaling of the target: positions and scores scale inversely,
# so an absolute floor would swamp whichever side has the smaller scale
# (e.g. the score covariance of a very wide target). Note that the exact
# affine equivariance of the unregularized Fisher equation is only
# preserved under orthogonal and scalar transformations by this scalar
# ridge, not under arbitrary affine maps:
_rel_regularization(γ::Real, C::AbstractMatrix{T}) where {T<:AbstractFloat} =
    T(γ) * max(tr(C) / size(C, 1), floatmin(T))
_rel_regularization(γ::Real, var_vec::AbstractVector{T}) where {T<:AbstractFloat} =
    T(γ) * max(sum(var_vec) / length(var_vec), floatmin(T))

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

function _fisher_diagonal_geometry(acc::_XGMoments, γ::Real)
    n = acc.n
    var_x_raw = acc.M2_x ./ (n - 1)
    var_g_raw = acc.M2_g ./ (n - 1)
    var_x = var_x_raw .+ _rel_regularization(γ, var_x_raw)
    var_g = var_g_raw .+ _rel_regularization(γ, var_g_raw)
    g = sqrt.(var_x ./ var_g)
    μ = acc.mean_x .+ g .* acc.mean_g
    return Diagonal(g), μ
end

_fisher_geometry(::DiagonalFisherEstimator, acc::_XGMoments, γ::Real) =
    _fisher_diagonal_geometry(acc, γ)

_fisher_geometry(::LowRankFisherEstimator, acc::_XGMoments, γ::Real) =
    _fisher_diagonal_geometry(acc, γ)

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

# Fit one projected low-rank correction from an immutable block. The
# diagonal base comes from the separate streaming estimator.
function _fit_lowrank_candidate(
    est::LowRankFisherEstimator,
    dvec::AbstractVector,
    Xfit::AbstractMatrix,
    Gfit::AbstractMatrix,
    γ::Real,
)
    m = size(Xfit, 2)
    T = float(promote_type(eltype(dvec), eltype(Xfit), eltype(Gfit)))
    empty = _LowRankCandidate(
        T[],
        zeros(T, length(dvec), 0),
        zeros(T, length(dvec), 0),
        Symmetric(zeros(T, 0, 0)),
    )
    size(Gfit) == size(Xfit) || throw(DimensionMismatch(
        "position and score fit blocks must have the same shape",
    ))
    length(dvec) == size(Xfit, 1) || throw(DimensionMismatch(
        "the diagonal base and fit block dimensions must match",
    ))
    m >= 8 || return empty
    all(x -> isfinite(x) && x > 0, dvec) || return empty
    all(isfinite, Xfit) && all(isfinite, Gfit) || return empty
    dsq = sqrt.(dvec)

    Xc = (Xfit .- (sum(Xfit, dims = 2) ./ m)) ./ dsq
    Gc = (Gfit .- (sum(Gfit, dims = 2) ./ m)) .* dsq
    all(isfinite, Xc) && all(isfinite, Gc) || return empty
    Q = Matrix(qr(hcat(Xc, Gc)).Q)
    Px = Q' * Xc
    Pg = Q' * Gc
    Cx_q_raw = Symmetric(Px * Px' ./ (m - 1))
    Cg_q_raw = Symmetric(Pg * Pg' ./ (m - 1))
    all(isfinite, Cx_q_raw) && all(isfinite, Cg_q_raw) || return empty
    Cx_q = Symmetric(Cx_q_raw + _rel_regularization(γ, Cx_q_raw) * I)
    Cg_q = Symmetric(Cg_q_raw + _rel_regularization(γ, Cg_q_raw) * I)
    Mq = _spd_riccati_solve(Cx_q, Cg_q)
    all(isfinite, Mq) || return empty
    E = eigen(Symmetric(Matrix(Mq)))
    λ, V = E.values, Q * E.vectors
    W, S, λ_kept, V_kept =
        _lowrank_correction(dsq, λ, V, est.cutoff, est.max_rank)
    all(isfinite, λ_kept) && all(isfinite, V_kept) &&
        all(isfinite, W) && all(isfinite, S) || return empty
    return _LowRankCandidate(λ_kept, V_kept, W, S)
end

_valid_lowrank_baseline(dvec, mu) =
    all(x -> isfinite(x) && x > 0, dvec) && all(isfinite, mu)

_valid_lowrank_spectrum(candidate::_LowRankCandidate) =
    length(candidate.lambda) == 1 &&
    all(x -> isfinite(x) && 0.1 <= x <= 20, candidate.lambda)

function _valid_lowrank_candidate(
    candidate::_LowRankCandidate,
    dvec,
    mu,
)
    _valid_lowrank_baseline(dvec, mu) || return false
    _valid_lowrank_spectrum(candidate) || return false
    all(isfinite, candidate.vectors) || return false
    all(isfinite, candidate.W) && all(isfinite, candidate.S) || return false
    G = _lowrank_geometry(dvec, candidate)
    G_dense = Symmetric(Matrix(G))
    return all(isfinite, G_dense) && isposdef(G_dense)
end

_lowrank_geometry(dvec::AbstractVector, candidate::_LowRankCandidate) =
    woodbury_operator(Diagonal(dvec), candidate.W, candidate.S)

_lowrank_geometry_diagonal(dvec::AbstractVector, candidate::_LowRankCandidate) =
    dvec .+ vec(sum((candidate.W * candidate.S) .* candidate.W, dims = 2))

function _fisher_loss(A, mu, x, alpha)
    fisher_residual = A' * alpha + A \ (x - mu)
    return sum(abs2, fisher_residual)
end

function _invalid_lowrank_validation_stats(::Type{T} = Float64) where {T<:AbstractFloat}
    return (
        mean = T(NaN),
        se = T(NaN),
        se_within = T(NaN),
        se_between = T(NaN),
        n_eff = zero(T),
        valid = false,
    )
end

function _lowrank_validation_stats(delta::AbstractMatrix)
    T = float(eltype(delta))
    n_walkers, n_steps = size(delta)
    n_walkers > 0 && n_steps > 1 || return _invalid_lowrank_validation_stats(T)
    all(isfinite, delta) || return _invalid_lowrank_validation_stats(T)

    walker_means = vec(mean(delta, dims = 2))
    gamma0 = zeros(T, n_walkers)
    sigma2_lr = zeros(T, n_walkers)
    for walker in axes(delta, 1)
        trace = view(delta, walker, :)
        centered = trace .- walker_means[walker]
        gamma0[walker] = sum(abs2, centered) / n_steps
        gamma0[walker] > 0 || return _invalid_lowrank_validation_stats(T)
        tau = tau_int_from_atc(fft_autocor(trace), GeyerAutocorLen())
        isfinite(tau) || return _invalid_lowrank_validation_stats(T)
        sigma2_lr[walker] = gamma0[walker] * max(tau, one(tau))
    end

    se_within2 = sum(sigma2_lr) / (n_walkers^2 * n_steps)
    se_between2 = n_walkers >= 2 ? var(walker_means) / n_walkers : zero(T)
    se2 = max(se_within2, se_between2)
    se2 > 0 && isfinite(se2) || return _invalid_lowrank_validation_stats(T)
    n_eff = min(T(n_walkers * n_steps), mean(gamma0) / se2)
    isfinite(n_eff) || return _invalid_lowrank_validation_stats(T)

    return (
        mean = mean(walker_means),
        se = sqrt(se2),
        se_within = sqrt(se_within2),
        se_between = sqrt(se_between2),
        n_eff,
        valid = true,
    )
end

function _lowrank_loss_improves(
    delta::AbstractMatrix;
    alpha::Real = 0.01,
    min_n_eff::Real = 20,
)
    stats = _lowrank_validation_stats(delta)
    stats.valid || return false
    stats.n_eff >= min_n_eff || return false
    z = quantile(Normal(), 1 - alpha)
    return stats.mean - z * stats.se > 0
end

function _lowrank_validation_accepts(
    candidate::_LowRankCandidate,
    delta_baseline::AbstractMatrix,
    delta_offdiag::AbstractMatrix;
    kwargs...,
)
    _valid_lowrank_spectrum(candidate) || return false
    return _lowrank_loss_improves(delta_baseline; kwargs...) &&
        _lowrank_loss_improves(delta_offdiag; kwargs...)
end

function _lowrank_validation_factors(est, campaign::_LowRankCampaign)
    G1 = _lowrank_geometry(campaign.baseline_dvec, campaign.candidate)
    return (
        _fisher_A(est, Diagonal(campaign.baseline_dvec)),
        _fisher_A(est, G1),
        _fisher_A(
            est,
            Diagonal(_lowrank_geometry_diagonal(
                campaign.baseline_dvec,
                campaign.candidate,
            )),
        ),
    )
end

function _fit_lowrank_campaign!(
    campaign::_LowRankCampaign,
    est::LowRankFisherEstimator,
    regularization::Real,
    is_mala::Bool,
    f_transform,
)
    candidate = _fit_lowrank_candidate(
        LowRankFisherEstimator(est.cutoff, 1),
        campaign.baseline_dvec,
        campaign.fit.X,
        campaign.fit.G,
        regularization,
    )
    campaign.candidate = _valid_lowrank_candidate(
        candidate,
        campaign.baseline_dvec,
        campaign.baseline_mu,
    ) ? candidate : nothing
    if is_mala && !isnothing(campaign.candidate)
        G = _lowrank_geometry(campaign.baseline_dvec, campaign.candidate)
        return MulAdd(
            _fisher_A(est, G),
            oftype(f_transform.b, campaign.baseline_mu),
        )
    end
    return nothing
end

function _decide_lowrank_campaign(
    campaign::_LowRankCampaign,
    est::LowRankFisherEstimator,
    is_mala::Bool,
    f_transform,
)
    candidate = campaign.candidate
    campaign.attempted = true
    campaign.phase = _LRFrozen
    admitted = !isnothing(candidate) && _lowrank_validation_accepts(
        candidate,
        campaign.validation_loss,
        campaign.validation_offdiag_loss,
    )
    if admitted
        campaign.admitted = true
        if !is_mala
            G = _lowrank_geometry(campaign.baseline_dvec, candidate)
            return MulAdd(
                _fisher_A(est, G),
                oftype(f_transform.b, campaign.baseline_mu),
            )
        end
    elseif is_mala && !isnothing(candidate)
        return MulAdd(
            _fisher_A(est, Diagonal(campaign.baseline_dvec)),
            oftype(f_transform.b, campaign.baseline_mu),
        )
    end
    return f_transform
end

_dense_spd(G::Symmetric) = G
_dense_spd(G::Diagonal) = Symmetric(Matrix(G))
_dense_spd(G) = Symmetric(Matrix(G))

# The matrix part of the transform to commit, in the structure the
# estimator maintains:
_fisher_A(::DenseFisherEstimator, G) = cholesky(Positive, Matrix(_dense_spd(G))).L
_fisher_A(::DiagonalFisherEstimator, G::Diagonal) = Diagonal(sqrt.(G.diag))
function _fisher_A(::LowRankFisherEstimator, G::Diagonal)
    n_dims = length(G.diag)
    T = eltype(G.diag)
    return _lowrank_gram_factor(G.diag, zeros(T, n_dims, 0), Symmetric(zeros(T, 0, 0)))
end
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
# sqrt(2 n_dims / n_observations), with the effective (autocorrelation-
# corrected) observation count, so the effective commit threshold stays
# above it:
function _effective_commit_threshold(sched::DriftCommitSchedule, n_dims::Integer, n_obs::Real)
    return sched.commit_threshold + 3 * sqrt(2 * n_dims / max(n_obs, 1))
end

function _diagonal_drift(old::AbstractVector, new::AbstractVector)
    tiny = floatmin(float(promote_type(eltype(old), eltype(new))))
    return sqrt(sum(x -> abs2(log(max(x, tiny))), new ./ old))
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

    est = tuner_state.estimator
    campaign = tuner_state.campaign
    phase = nothing
    if est isa LowRankFisherEstimator && !isnothing(campaign)
        _advance_lowrank_campaign!(campaign, campaign.cycle_step + 1)
        phase = campaign.phase
    end

    A = f_transform.A
    xs_prop = proposed.x.v
    xs_curr = current.x.v
    accepted = chain_state.accepted
    is_mala = proposal isa MALAProposalState
    validation_idx = if phase == _LRValidate
        validation_start = campaign.fit_start + campaign.fit_steps + campaign.guard_steps
        campaign.cycle_step - validation_start + 1
    else
        0
    end

    validating = phase == _LRValidate && !isnothing(campaign.candidate)
    validation_factors = validating ?
        _lowrank_validation_factors(est, campaign) : (nothing, nothing, nothing)
    baseline_A, candidate_A, candidate_diag_A = validation_factors

    for i in eachindex(xs_prop, z_grads)
        # Score transport into the fixed pre-adaptive space: for x = A z + μ
        # the pulled-back gradient is β = Aᵀ α, so α = A⁻ᵀ β:
        α = A' \ z_grads[i]
        x_i = accepted[i] ? xs_prop[i] : xs_curr[i]

        if phase == _LRFit
            column = campaign.fit.nsteps * length(xs_prop) + i
            campaign.fit.X[:, column] .= x_i
            campaign.fit.G[:, column] .= α
        elseif validating
            loss0 = _fisher_loss(baseline_A, campaign.baseline_mu, x_i, α)
            loss1 = _fisher_loss(candidate_A, campaign.baseline_mu, x_i, α)
            campaign.validation_loss[i, validation_idx] = loss0 - loss1
            loss_diag = _fisher_loss(
                candidate_diag_A,
                campaign.baseline_mu,
                x_i,
                α,
            )
            campaign.validation_offdiag_loss[i, validation_idx] =
                loss_diag - loss1
        elseif isnothing(campaign) || phase == _LRWaiting
            # The gradients refer to the selected states, so the positions
            # must too. Repeated states remain valid observations.
            _moments_update!(tuner_state.acc_a, x_i, α)
            _moments_update!(tuner_state.acc_b, x_i, α)
        end
    end

    provisional_transform = nothing
    if phase == _LRFit
        campaign.fit.nsteps += 1
        if campaign.fit.nsteps == campaign.fit_steps
            provisional_transform = _fit_lowrank_campaign!(
                campaign,
                est,
                tuner_state.tuning.regularization,
                is_mala,
                f_transform,
            )
        end
    end

    tuner_state.nsteps += 1
    accumulating = isnothing(campaign) || phase == _LRWaiting
    if accumulating && tuner_state.nsteps % tuner_state.memory_length == 0
        tuner_state.acc_a = tuner_state.acc_b
        tuner_state.acc_b = _new_moments(
            tuner_state.estimator,
            tuner_state.acc_b.mean_x,
            tuner_state.acc_b.lag1.stride,
        )
    end

    if !isnothing(campaign) && phase == _LRWaiting &&
            campaign.cycle_step == campaign.fit_start - 1
        G, μ = _fisher_diagonal_geometry(
            tuner_state.acc_a,
            tuner_state.tuning.regularization,
        )
        if !_valid_lowrank_baseline(diag(G), μ)
            campaign.phase = _LRFrozen
            campaign.attempted = true
            return f_transform, tuner_state, chain_state
        end
        campaign.baseline_dvec = copy(diag(G))
        campaign.baseline_mu = copy(μ)
        tuner_state.committed_diag = copy(diag(G))
        A_new = _fisher_A(est, G)
        return MulAdd(A_new, oftype(f_transform.b, μ)), tuner_state, chain_state
    end

    if !isnothing(provisional_transform)
        return provisional_transform, tuner_state, chain_state
    end

    if phase == _LRValidate && validation_idx == campaign.validation_steps
        f_transform_new = _decide_lowrank_campaign(
            campaign,
            est,
            is_mala,
            f_transform,
        )
        return f_transform_new, tuner_state, chain_state
    end

    if !isnothing(campaign) && phase != _LRWaiting
        return f_transform, tuner_state, chain_state
    end

    sched = tuner_state.tuning.schedule
    acc = tuner_state.acc_a
    if tuner_state.nsteps % sched.check_interval == 0 && acc.n >= tuner_state.min_observations
        local G, μ, drift
        if est isa LowRankFisherEstimator
            G, μ = _fisher_diagonal_geometry(acc, tuner_state.tuning.regularization)
            dvec = diag(G)
            drift = isnothing(tuner_state.committed_diag) ?
                oftype(sched.commit_threshold, Inf) :
                _diagonal_drift(tuner_state.committed_diag, dvec)
        else
            G, μ = _fisher_geometry(est, acc, tuner_state.tuning.regularization)
            drift = _transform_drift(A, G)
        end
        if drift > _effective_commit_threshold(sched, tuner_state.n_dims, _effective_nobs(acc))
            if est isa LowRankFisherEstimator
                tuner_state.committed_diag = copy(diag(G))
            end
            A_new = _fisher_A(est, G)
            b_new = oftype(f_transform.b, μ)
            return MulAdd(A_new, b_new), tuner_state, chain_state
        end
    end

    return f_transform, tuner_state, chain_state
end


# Gradient-based proposals default to Fisher-divergence transform tuning
# for all affine transform structures. The TriangularAffineTransform
# method only resolves the ambiguity against the RAMTuning default, which
# covers that structure for gradient-free proposals:
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::Union{HamiltonianMC,MALAProposal}, ::TriangularAffineTransform) = FisherTransformTuning()
bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, ::Union{HamiltonianMC,MALAProposal}, ::AbstractAffineTransform) = FisherTransformTuning()

function bat_default(::Type{TransformedMCMC}, ::Val{:transform_tuning}, proposal::MCMCProposal, at::AbstractAffineTransform)
    throw(ArgumentError("Tuning a $(nameof(typeof(at))) currently requires a gradient-based MCMC proposal (like HamiltonianMC or MALAProposal), not $(nameof(typeof(proposal)))"))
end
