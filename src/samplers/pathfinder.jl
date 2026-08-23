# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Single-path implementation of the Pathfinder algorithm (L. Zhang,
# B. Carpenter, A. Gelman, A. Vehtari, "Pathfinder: Parallel quasi-Newton
# variational inference", JMLR 23(306), 2022,
# https://jmlr.org/papers/v23/21-0889.html), reduced to what is needed to
# seed MCMC space transformations: the mean and covariance of the best local
# Gaussian approximation along an L-BFGS trajectory. The trajectory comes
# from a `maximize_density` backend with trace recording, the
# inverse-Hessian reconstruction and factorization follow the reference
# implementation Pathfinder.jl (MIT License, Copyright (c) 2021 Seth Axen
# and contributors).


# All array operations here are expressed as broadcasts, reductions and
# matrix products over whole arrays, never as elementwise indexing, so
# that the estimates follow the storage of the trajectory instead of
# being tied to the CPU.


# Copies the columns of a ring buffer into chronological order. Two
# contiguous column ranges instead of a gather with an index vector,
# which would force the subsequent matrix products onto an elementwise
# fallback:
function _ordered_history!(dest::AbstractMatrix, src::AbstractMatrix, history_ind::Integer, J::Integer)
    ntail = J - history_ind
    copyto!(view(dest, :, 1:ntail), view(src, :, (history_ind + 1):J))
    copyto!(view(dest, :, (ntail + 1):J), view(src, :, 1:history_ind))
    return dest
end

# In-place triangular masks and symmetrization. The `LinearAlgebra`
# equivalents (`triu!`, `tril!`, `copytri!`) walk the matrix elementwise,
# these are single fused broadcasts. The matrix passed to `_symmetrize!`
# is symmetric up to rounding already, so averaging with the transpose is
# equivalent to copying one triangle onto the other:
_triu!(M::AbstractMatrix) = (M .= ifelse.(axes(M, 1) .<= axes(M, 2)', M, zero(eltype(M))); M)
_tril!(M::AbstractMatrix) = (M .= ifelse.(axes(M, 1) .>= axes(M, 2)', M, zero(eltype(M))); M)

function _symmetrize!(M::AbstractMatrix)
    Mᵀ = copy(transpose(M))
    M .= (M .+ Mᵀ) ./ 2
    return M
end


# Diagonal inverse-Hessian estimate, eq. 4.9 of Gilbert & Lemaréchal,
# "Some numerical experiments with variable-storage quasi-Newton algorithms",
# Mathematical Programming 45 (1989), https://doi.org/10.1007/BF01589113:
function _gilbert_init(α, s, y)
    a = sum(α .* y .* y)   # yᵀ Diagonal(α) y
    b = dot(y, s)
    c = sum(s .* s ./ α)   # sᵀ Diagonal(α)⁻¹ s
    return @. b / (a / α + y^2 - (a / c) * (s / α)^2)
end

# Compact representation H = Diagonal(α) + B * D * Bᵀ of the L-BFGS inverse
# Hessian estimate (theorem 2.2 of Byrd, Nocedal & Schnabel, "Representations
# of quasi-Newton matrices and their use in limited memory methods",
# Mathematical Programming 63, 1994, https://doi.org/10.1007/BF01582063):
function _lbfgs_inverse_hessian(α::AbstractVector, S0::AbstractMatrix, Y0::AbstractMatrix, history_ind::Integer, history_length::Integer)
    J = history_length
    n = size(α, 1)
    B = similar(α, n, 2J)
    D = fill!(similar(α, 2J, 2J), false)
    iszero(J) && return (α = copy(α), B = B, D = D)

    S = _ordered_history!(similar(α, n, J), S0, history_ind, J)
    Y = _ordered_history!(similar(α, n, J), Y0, history_ind, J)
    ΛY = α .* Y   # Diagonal(α) * Y

    @views begin
        B[:, 1:J] .= ΛY
        B[:, (J + 1):(2J)] .= S
    end

    # The blocks are built as standalone arrays and only copied into `D`
    # at the end. Backends specialize their matrix kernels on their own
    # array type, but not necessarily on views into it, so running the
    # triangular solve and products on slices of `D` would risk a fallback
    # to a host implementation:
    SᵀY = S' * Y
    d = diag(SᵀY)
    R = UpperTriangular(_triu!(SᵀY))

    nR⁻¹ = fill!(similar(α, J, J), zero(eltype(α)))
    nR⁻¹[diagind(nR⁻¹)] .= -one(eltype(α))
    ldiv!(R, nR⁻¹)
    nR⁻ᵀ = copy(transpose(nR⁻¹))

    # Diagonal(diag(SᵀY)) + Yᵀ Diagonal(α) Y, congruence-transformed by
    # -R⁻¹ (symmetric by construction, up to rounding). The products are
    # out-of-place: in-place `rmul!`/`lmul!` alias input and output, which
    # a parallel backend cannot honor:
    E0 = Y' * ΛY
    E0[diagind(E0)] .+= d
    _symmetrize!(E0)
    E = LowerTriangular(nR⁻ᵀ) * E0 * UpperTriangular(nR⁻¹)

    @views begin
        D[1:J, (J + 1):(2J)] .= nR⁻¹
        D[(J + 1):(2J), 1:J] .= nR⁻ᵀ
        D[(J + 1):(2J), (J + 1):(2J)] .= E
    end

    return (α = copy(α), B = B, D = D)
end

# One curvature-pair update of the L-BFGS ring buffers and the diagonal
# estimate, returns the updated (α, history_ind, history_length_effective):
function _lbfgs_curvature_update!(
    S::AbstractMatrix, Y::AbstractMatrix, α::AbstractVector,
    history_ind::Integer, history_length_effective::Integer,
    s::AbstractVector, y::AbstractVector, history_length::Integer, ϵ::Real
)
    if dot(y, s) > ϵ * sum(abs2, y)  # positive curvature, safe to update
        history_ind = mod1(history_ind + 1, history_length)
        history_length_effective = max(history_ind, history_length_effective)
        S[:, history_ind] .= s
        Y[:, history_ind] .= y
        α = _gilbert_init(α, s, y)
    end
    return α, history_ind, history_length_effective
end

# Inverse-Hessian estimates along an L-BFGS trajectory of positions θs with
# log-density gradients ∇logpθs. Materializes all estimates at once - the
# streaming fit below walks the trajectory with a single live estimate
# instead:
function _lbfgs_inverse_hessians(
    θs::AbstractVector{<:AbstractVector}, ∇logpθs::AbstractVector{<:AbstractVector};
    history_length::Integer = 6, ϵ::Real = 1e-12
)
    L = length(θs) - 1
    θ = θs[1]
    ∇logpθ = ∇logpθs[1]
    n = length(θ)

    history_ind = 0
    history_length_effective = 0
    s = similar(θ)
    y = similar(∇logpθ)
    S = similar(s, n, min(history_length, L))
    Y = similar(y, n, min(history_length, L))
    α = fill!(similar(θ), true)
    Hs = [_lbfgs_inverse_hessian(α, S, Y, history_ind, history_length_effective)]

    for l in 1:L
        θlp1, ∇logpθlp1 = θs[l + 1], ∇logpθs[l + 1]
        s .= θlp1 .- θ
        y .= ∇logpθ .- ∇logpθlp1
        α, history_ind, history_length_effective =
            _lbfgs_curvature_update!(S, Y, α, history_ind, history_length_effective, s, y, history_length, ϵ)
        θ, ∇logpθ = θlp1, ∇logpθlp1
        push!(Hs, _lbfgs_inverse_hessian(α, S, Y, history_ind, history_length_effective))
    end

    return Hs
end


"""
    BAT.pathfinder_gaussian_fit(
        f_logd::Function, x0::AbstractVector{<:Real},
        optalg, context::BATContext;
        history_length::Integer = 6, ndraws_elbo::Integer = 5
    )

*BAT-internal, not part of stable public API.*

Runs single-path Pathfinder ([Zhang et al.
(2022)](https://jmlr.org/papers/v23/21-0889.html)) from `x0` and returns the
mean `μ`, dense covariance `Σ` and `elbo` of the maximum-ELBO local Gaussian
approximation of the target along an L-BFGS trajectory, as a `NamedTuple`,
or `nothing` if no approximation with a finite ELBO is found.

The trajectory is generated by maximizing the log-density function `f_logd`
via [`maximize_density`](@ref) with `optalg`, which must be a gradient-based
backend that records iterates and gradients (e.g. an [`OptimAlg`](@ref)
with `Optim.LBFGS`). Optimizer failures count as path-local failures.
"""
function pathfinder_gaussian_fit(
    f_logd::Function, x0::AbstractVector{<:Real}, optalg, context::BATContext;
    history_length::Integer = 6, ndraws_elbo::Integer = 5
)
    @argcheck history_length >= 1
    @argcheck ndraws_elbo >= 1

    traced_optalg = @set optalg.store_trace = true
    r = try
        maximize_density(f_logd, x0, traced_optalg, context)
    catch err
        err isa InterruptException && rethrow()
        @warn "Pathfinder path failed, skipping this start point" exception = err
        return nothing
    end

    if isnothing(r.trace) || !haskey(r.trace, :grad_logd)
        throw(ArgumentError(
            "Pathfinder requires an optimization backend that records iterates and gradients, like OptimAlg with a first-order Optim optimizer"
        ))
    end

    # A non-finite start leaves the optimizer nowhere to go; this usually
    # indicates bad initial values, so it warrants a warning even though
    # it is only a path-local failure:
    if isempty(r.trace.logd) || !isfinite(first(r.trace.logd))
        @warn "Pathfinder can't start from a point with non-finite log-density, skipping this start point"
        return nothing
    end

    xs, grads = r.trace.v, r.trace.grad_logd
    gen = get_gencontext(context)
    rng = get_rng(gen)
    cunit = get_compute_unit(gen)

    # A path with no accepted L-BFGS step carries no curvature information;
    # its only candidate would be the arbitrary initial inverse-Hessian
    # estimate, so the path fails instead:
    length(xs) > 1 || return nothing

    T = eltype(first(xs))
    n = length(first(xs))
    L = length(xs) - 1

    # The inverse-Hessian candidates are streamed along the trajectory:
    # only the current candidate and the running best are live
    # (materializing all L candidates at once would need O(L n history)
    # memory). The initial estimate H₀ = I carries no curvature
    # information from the path and never enters the ELBO competition
    # (reference Pathfinder excludes it as well):
    history_ind = 0
    history_length_effective = 0
    s = similar(xs[1])
    y = similar(xs[1])
    S = similar(s, n, min(history_length, L))
    Y = similar(y, n, min(history_length, L))
    α = fill!(similar(xs[1]), true)

    best_elbo = T(-Inf)
    best = nothing

    for l in 1:L
        s .= xs[l + 1] .- xs[l]
        y .= grads[l] .- grads[l + 1]
        α, history_ind, history_length_effective =
            _lbfgs_curvature_update!(S, Y, α, history_ind, history_length_effective, s, y, history_length, 1e-12)
        Hl = _lbfgs_inverse_hessian(α, S, Y, history_ind, history_length_effective)

        # The inverse-Hessian estimate as a Woodbury-structured operator
        # (D is symmetric by construction, up to rounding):
        H = woodbury_operator(Diagonal(Hl.α), Hl.B, Symmetric(Hl.D))

        # Draws are generated on the CPU and adapted to the compute unit,
        # as elsewhere in BAT (the context RNG is CPU-side):
        Z = adapt(cunit, randn(rng, T, n, ndraws_elbo))

        elbo, μ = _pathfinder_elbo(f_logd, H, xs[l + 1], grads[l + 1], Z, T)
        if elbo > best_elbo
            best_elbo = elbo
            best = (μ = μ, α = Hl.α, B = Hl.B, D = Hl.D)
        end
    end

    isnothing(best) && return nothing
    return (μ = best.μ, Σ = _woodbury_matrix(best.α, best.B, best.D), elbo = best_elbo)
end


# ELBO of the local Gaussian approximation induced by an inverse-Hessian
# estimate `H` at the trajectory point `x`, together with that
# approximation's mean. Unusable candidates score `-Inf` rather than
# raising or being skipped, so that candidate selection is a reduction
# over scores instead of data-dependent control flow:
function _pathfinder_elbo(f_logd, H, x::AbstractVector, grad::AbstractVector, Z::AbstractMatrix, ::Type{T}) where {T<:Real}
    n = size(Z, 1)
    μ = x .+ H * grad
    all(isfinite, μ) || return T(-Inf), μ

    # The stable "square root" factorization of Zhang et al. (2022),
    # appendix A, which comes with a structural log-determinant. It
    # rejects indefinite estimates by raising, so this is the one spot
    # that still needs a value-dependent branch.
    # ToDo: Use a non-raising factorization entry point once
    # MatrixShapedOperators offers one.
    F = try
        rowgram_factor(H)
    catch err
        err isa PosDefException || rethrow()
        return T(-Inf), μ
    end

    X = F * Z .+ μ
    znormsq = vec(sum(abs2, Z, dims = 1))
    logabsdet_L = first(logabsdet(F))
    logq = @. -(n * T(log2π) + znormsq) / 2 - logabsdet_L
    # ToDo: Evaluate the target on all draws at once once BAT supports
    # batched density evaluation.
    logd_mean = mean(T(f_logd(view(X, :, j))) for j in axes(X, 2))
    return logd_mean - T(mean(logq)), μ
end

# Dense covariance from the compact inverse-Hessian representation,
# materialized in the storage of its factors. Going through
# `AbstractMatrix(::MatrixShapedOperator)` instead would apply the
# operator to a CPU identity matrix:
function _woodbury_matrix(α::AbstractVector, B::AbstractMatrix, D::AbstractMatrix)
    Σ = B * (Symmetric(D) * transpose(B))
    Σ[diagind(Σ)] .+= α
    return Σ
end


function _affine_init_moments(tinit::PathfinderTransformInit, target::AbstractMeasure, v_init::AbstractVector, context::BATContext)
    # Pathfinder is gradient-based by definition, so a missing AD backend
    # is a configuration error, caught here before any path runs (the
    # per-path failure handling would otherwise swallow it):
    _ = get_valid_adselector(context, tinit)

    # Deliberately not checked_logdensityof: the optimizer's line search
    # probes far-out points where the target may return NaN, which counts
    # as a per-path failure instead of aborting the whole initialization:
    f_logd = logdensityof(target)

    fits = filter(!isnothing, [
        pathfinder_gaussian_fit(
            f_logd, x0, tinit.optalg, context,
            history_length = tinit.history_length,
            ndraws_elbo = tinit.ndraws_elbo
        )
        for x0 in v_init
    ])

    if isempty(fits)
        @warn "Pathfinder-based space transformation initialization failed, falling back to prior-based initialization"
        return _affine_init_moments(PriorApproxTransformInit(), target, v_init, context)
    end

    # A path that converged to a clearly inferior mode or shoulder would
    # contaminate the equal-weight moment-matched mixture, so fits whose
    # estimated ELBO is more than ten nats below the best are dropped - a
    # heuristic fit-quality threshold (ELBO differences are variational
    # fit scores, not mixture weights):
    elbo_best = maximum(fit.elbo for fit in fits)
    fits = [fit for fit in fits if fit.elbo >= elbo_best - 10]

    μ = mean(fit.μ for fit in fits)
    Σ = mean(fit.Σ for fit in fits)
    if length(fits) > 1
        # Moment-matched Gaussian-mixture covariance over the per-walker fits:
        Σ = Σ + cov(stack(fit.μ for fit in fits), dims = 2, corrected = false)
    end

    return Σ, μ
end
