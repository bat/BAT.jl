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


# Diagonal inverse-Hessian estimate, eq. 4.9 of Gilbert & Lemaréchal,
# "Some numerical experiments with variable-storage quasi-Newton algorithms",
# Mathematical Programming 45 (1989), https://doi.org/10.1007/BF01589113:
function _gilbert_init(α, s, y)
    a = dot(y, Diagonal(α), y)
    b = dot(y, s)
    c = dot(s, inv(Diagonal(α)), s)
    return @. b / (a / α + y^2 - (a / c) * (s / α)^2)
end

# Compact representation H = Diagonal(α) + B * D * Bᵀ of the L-BFGS inverse
# Hessian estimate (theorem 2.2 of Byrd, Nocedal & Schnabel, "Representations
# of quasi-Newton matrices and their use in limited memory methods",
# Mathematical Programming 63, 1994, https://doi.org/10.1007/BF01582063):
function _lbfgs_inverse_hessian(α::AbstractVector, S0::AbstractMatrix, Y0::AbstractMatrix, history_ind::Integer, history_length::Integer)
    J = history_length
    B = similar(α, size(α, 1), 2J)
    D = fill!(similar(α, 2J, 2J), false)
    iszero(J) && return (α = copy(α), B = B, D = D)

    hist_inds = [(history_ind + 1):history_length; 1:history_ind]
    @views begin
        S = S0[:, hist_inds]
        Y = Y0[:, hist_inds]
        B₁ = B[:, 1:J]
        B₂ = B[:, (J + 1):(2J)]
        D₁₂ = D[1:J, (J + 1):(2J)]
        D₂₁ = D[(J + 1):(2J), 1:J]
        D₂₂ = D[(J + 1):(2J), (J + 1):(2J)]
    end

    mul!(B₁, Diagonal(α), Y)
    copyto!(B₂, S)
    mul!(D₂₂, S', Y)
    triu!(D₂₂)
    R = UpperTriangular(D₂₂)
    nRinv = UpperTriangular(D₁₂)
    copyto!(nRinv, -I)
    ldiv!(R, nRinv)
    nRinv′ = LowerTriangular(copyto!(D₂₁, nRinv'))
    tril!(D₂₂) # eliminate all but the diagonal
    mul!(D₂₂, Y', B₁, true, true)
    LinearAlgebra.copytri!(D₂₂, 'U', false, false)
    rmul!(D₂₂, nRinv)
    lmul!(nRinv′, D₂₂)

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
    rng = get_rng(context)

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
        # (D is symmetric by construction, up to rounding); its stable
        # "square root" factorization (Zhang et al. 2022, appendix A)
        # comes with a structural log-determinant:
        H = woodbury_operator(Diagonal(Hl.α), Hl.B, Symmetric(Hl.D))
        F = try
            rowgram_factor(H)
        catch err
            err isa PosDefException || rethrow()
            continue
        end
        μ = xs[l + 1] .+ H * grads[l + 1]
        all(isfinite, μ) || continue

        Z = randn(rng, T, n, ndraws_elbo)
        znormsq = vec(sum(abs2, Z, dims = 1))
        X = F * Z .+ μ
        logabsdet_L = first(logabsdet(F))

        elbo = zero(T)
        for j in 1:ndraws_elbo
            logq_j = -(n * T(log2π) + znormsq[j]) / 2 - logabsdet_L
            elbo += (T(f_logd(view(X, :, j))) - logq_j) / ndraws_elbo
        end

        if elbo > best_elbo
            best_elbo = elbo
            best = (μ = μ, α = Hl.α, B = Hl.B, D = Hl.D)
        end
    end

    isnothing(best) && return nothing
    Σ = Matrix(woodbury_operator(Diagonal(best.α), best.B, Symmetric(best.D)))
    return (μ = best.μ, Σ = Σ, elbo = best_elbo)
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
