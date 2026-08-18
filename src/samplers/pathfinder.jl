# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Native single-path implementation of the Pathfinder algorithm (L. Zhang,
# B. Carpenter, A. Gelman, A. Vehtari, "Pathfinder: Parallel quasi-Newton
# variational inference", JMLR 23(306), 2022), reduced to what is needed to
# seed MCMC space transformations: the mean and covariance of the best local
# Gaussian approximation along an L-BFGS trajectory. The inverse-Hessian
# reconstruction and factorization follow the reference implementation
# Pathfinder.jl (MIT License, Copyright (c) 2021 Seth Axen and contributors).


# L-BFGS two-loop recursion, S/Y/invrho in minimization convention:
function _lbfgs_direction(g::AbstractVector{T}, S::AbstractVector, Y::AbstractVector, invrho::AbstractVector) where {T<:Real}
    q = copy(g)
    m = length(S)
    if m > 0
        a = Vector{T}(undef, m)
        for i in m:-1:1
            a[i] = dot(S[i], q) / invrho[i]
            q .-= a[i] .* Y[i]
        end
        q .*= invrho[m] / sum(abs2, Y[m])
        for i in 1:m
            b = dot(Y[i], q) / invrho[i]
            q .+= (a[i] - b) .* S[i]
        end
    end
    q .*= -1
    return q
end

# Maximizes the log-density given by `f_logdgrad(x) == (logd, grad)` via
# L-BFGS with backtracking line search. Returns the trace of iterates and
# their log-density gradients, or `nothing` if the starting point is
# unusable (a path-local failure, other start points may still succeed).
function _lbfgs_trace(
    f_logdgrad::Function, x0::AbstractVector{<:Real};
    maxiters::Integer, history_length::Integer,
    grad_tol::Real = 1e-8, max_backtracks::Integer = 40
)
    T = float(eltype(x0))
    x = convert(Vector{T}, x0)
    logd, grad_0 = f_logdgrad(x)
    grad = convert(Vector{T}, grad_0)
    if !(isfinite(logd) && all(isfinite, grad))
        @warn "Pathfinder can't start from a point with non-finite log-density or gradient, skipping this start point"
        return nothing
    end

    xs = [x]
    grads = [grad]
    S = Vector{Vector{T}}()
    Y = Vector{Vector{T}}()
    invrho = Vector{T}()

    f::T = -logd
    for iter in 1:maxiters
        maximum(abs, grad) > grad_tol || break

        g = -grad
        d = _lbfgs_direction(g, S, Y, invrho)
        dg = dot(d, g)
        if !(dg < 0)
            # Not a descent direction, reset to steepest descent:
            empty!(S); empty!(Y); empty!(invrho)
            d = -g
            dg = -sum(abs2, g)
        end

        # Backtracking line search with Armijo condition:
        t = iter == 1 ? min(one(T), inv(norm(d))) : one(T)
        x_new = similar(x)
        f_new::T = NaN
        grad_new = grad
        accepted = false
        for _ in 1:max_backtracks
            @. x_new = x + t * d
            logd_new, grad_new_0 = f_logdgrad(x_new)
            f_new = -logd_new
            if isfinite(f_new) && all(isfinite, grad_new_0) && f_new <= f + T(1//10^4) * t * dg
                grad_new = convert(Vector{T}, grad_new_0)
                accepted = true
                break
            end
            t /= 2
        end
        accepted || break

        s = x_new .- x
        y = grad .- grad_new
        if dot(y, s) > eps(T) * sum(abs2, y)
            push!(S, s); push!(Y, y); push!(invrho, dot(y, s))
            if length(S) > history_length
                popfirst!(S); popfirst!(Y); popfirst!(invrho)
            end
        end

        x, f, grad = x_new, f_new, grad_new
        push!(xs, x)
        push!(grads, grad)
    end

    return xs, grads
end


# Diagonal inverse-Hessian estimate, eq. 4.9 of Gilbert & Lemaréchal,
# "Some numerical experiments with variable-storage quasi-Newton algorithms",
# Mathematical Programming 45 (1989):
function _gilbert_init(α, s, y)
    a = dot(y, Diagonal(α), y)
    b = dot(y, s)
    c = dot(s, inv(Diagonal(α)), s)
    return @. b / (a / α + y^2 - (a / c) * (s / α)^2)
end

# Compact representation H = Diagonal(α) + B * D * Bᵀ of the L-BFGS inverse
# Hessian estimate (theorem 2.2 of Byrd, Nocedal & Schnabel, "Representations
# of quasi-Newton matrices and their use in limited memory methods",
# Mathematical Programming 63, 1994):
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

# Inverse-Hessian estimates along an L-BFGS trajectory of positions θs with
# log-density gradients ∇logpθs:
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
        if dot(y, s) > ϵ * sum(abs2, y)  # positive curvature, safe to update
            history_ind = mod1(history_ind + 1, history_length)
            history_length_effective = max(history_ind, history_length_effective)
            S[1:n, history_ind] .= s
            Y[1:n, history_ind] .= y
            α = _gilbert_init(α, s, y)
        end
        θ, ∇logpθ = θlp1, ∇logpθlp1
        push!(Hs, _lbfgs_inverse_hessian(α, S, Y, history_ind, history_length_effective))
    end

    return Hs
end


"""
    BAT.pathfinder_gaussian_fit(
        rng::AbstractRNG, f_logd::Function, f_logdgrad::Function,
        x0::AbstractVector{<:Real};
        maxiters::Integer = 1000, history_length::Integer = 6,
        ndraws_elbo::Integer = 5
    )

*BAT-internal, not part of stable public API.*

Runs single-path Pathfinder (Zhang et al. 2022) from `x0` and returns the
mean `μ`, dense covariance `Σ` and `elbo` of the maximum-ELBO local Gaussian
approximation of the target along the L-BFGS trajectory, as a `NamedTuple`,
or `nothing` if no approximation with a finite ELBO is found.

`f_logdgrad(x)` must return the tuple `(logd, grad)` of the target
log-density and its gradient, `f_logd(x)` just the log-density.
"""
function pathfinder_gaussian_fit(
    rng::AbstractRNG, f_logd::Function, f_logdgrad::Function, x0::AbstractVector{<:Real};
    maxiters::Integer = 1000, history_length::Integer = 6, ndraws_elbo::Integer = 5
)
    @argcheck maxiters >= 1
    @argcheck history_length >= 1
    @argcheck ndraws_elbo >= 1

    trace = _lbfgs_trace(f_logdgrad, x0, maxiters = maxiters, history_length = history_length)
    isnothing(trace) && return nothing
    xs, grads = trace

    # A path with no accepted L-BFGS step carries no curvature information;
    # its only candidate would be the arbitrary initial inverse-Hessian
    # estimate (see below), so the path fails instead:
    length(xs) > 1 || return nothing

    Hs = _lbfgs_inverse_hessians(xs, grads, history_length = history_length)

    T = eltype(first(xs))
    n = length(first(xs))
    best_elbo = T(-Inf)
    best = nothing

    # The first entry of Hs is the initial estimate H₀ = I, which contains
    # no curvature information from the path and must never win the ELBO
    # competition (reference Pathfinder excludes it as well):
    for l in Iterators.drop(eachindex(Hs), 1)
        (; α, B, D) = Hs[l]
        # The inverse-Hessian estimate as a Woodbury-structured operator
        # (D is symmetric by construction, up to rounding); its stable
        # "square root" factorization (Zhang et al. 2022, appendix A)
        # comes with a structural log-determinant:
        H = woodbury_operator(Diagonal(α), B, Symmetric(D))
        F = try
            rowgram_factor(H)
        catch err
            err isa PosDefException || rethrow()
            continue
        end
        μ = xs[l] .+ H * grads[l]
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
            best = (μ = μ, α = α, B = B, D = D)
        end
    end

    isnothing(best) && return nothing
    Σ = Matrix(woodbury_operator(Diagonal(best.α), best.B, Symmetric(best.D)))
    return (μ = best.μ, Σ = Σ, elbo = best_elbo)
end


function _init_affine_transform(tinit::PathfinderTransformInit, target::AbstractMeasure, v_init::AbstractVector, context::BATContext)
    adsel = get_valid_adselector(context, tinit)
    f_logd = checked_logdensityof(target)
    f_logdgrad = valgrad_func(f_logd, adsel)
    rng = get_rng(context)

    fits = filter(!isnothing, [
        pathfinder_gaussian_fit(
            rng, logdensityof(target), f_logdgrad, x0,
            maxiters = tinit.maxiters, history_length = tinit.history_length,
            ndraws_elbo = tinit.ndraws_elbo
        )
        for x0 in v_init
    ])

    if isempty(fits)
        @warn "Pathfinder-based space transformation initialization failed, falling back to prior-based initialization"
        return _init_affine_transform(PriorApproxTransformInit(), target, v_init, context)
    end

    μ = mean(fit.μ for fit in fits)
    Σ = mean(fit.Σ for fit in fits)
    if length(fits) > 1
        # Moment-matched Gaussian-mixture covariance over the per-walker fits:
        Σ = Σ + cov(stack(fit.μ for fit in fits), dims = 2, corrected = false)
    end

    A = cholesky(Positive, Σ).L
    return MulAdd(A, μ)
end
