# Code imported from Pathfinder.jl (https://github.com/sethaxen/Pathfinder.jl)
# under MIT License.
# Copyright (c) 2021 Seth Axen <seth.axen@gmail.com> and contributors

# ToDo: Remove as soon as available from a lightweight source.

# eq 4.9
# Gilbert, J.C., Lemaréchal, C. Some numerical experiments with variable-storage quasi-Newton algorithms.
# Mathematical Programming 45, 407–435 (1989). https://doi.org/10.1007/BF01589113
function gilbert_init(α, s, y)
    a = dot(y, Diagonal(α), y)
    b = dot(y, s)
    c = dot(s, inv(Diagonal(α)), s)
    return @. b / (a / α + y^2 - (a / c) * (s / α)^2)
end

function lbfgs_inverse_hessians(θs, ∇logpθs; Hinit=gilbert_init, history_length=5, ϵ=1e-12)
    L = length(θs) - 1
    θ = θs[1]
    ∇logpθ = ∇logpθs[1]
    n = length(θ)

    # allocate caches/containers
    history_ind = 0 # index of last set history entry
    history_length_effective = 0 # length of history so far
    s = similar(θ) # cache for BFGS update, i.e. sₗ = θₗ₊₁ - θₗ = -λ Hₗ ∇logpθₗ
    y = similar(∇logpθ) # cache for yₗ = ∇logpθₗ₊₁ - ∇logpθₗ = Hₗ₊₁ \ s₁ (secant equation)
    S = similar(s, n, min(history_length, L)) # history of s
    Y = similar(y, n, min(history_length, L)) # history of y
    α = fill!(similar(θ), true) # diag(H₀)
    H = lbfgs_inverse_hessian(Diagonal(α), S, Y, history_ind, history_length_effective) # H₀ = I
    Hs = [H] # trace of H

    for l in 1:L
        θlp1, ∇logpθlp1 = θs[l + 1], ∇logpθs[l + 1]
        s .= θlp1 .- θ
        y .= ∇logpθ .- ∇logpθlp1
        if dot(y, s) > ϵ * sum(abs2, y)  # curvature is positive, safe to update inverse Hessian
            # add s and y to history
            history_ind = mod1(history_ind + 1, history_length)
            history_length_effective = max(history_ind, history_length_effective)
            S[1:n, history_ind] .= s
            Y[1:n, history_ind] .= y

            # initial diagonal estimate of H
            α = Hinit(α, s, y)
        else
            @debug "Skipping inverse Hessian update from iteration $l to avoid negative curvature."
        end

        θ, ∇logpθ = θlp1, ∇logpθlp1
        H = lbfgs_inverse_hessian(Diagonal(α), S, Y, history_ind, history_length_effective)
        push!(Hs, H)
    end

    return Hs
end

function lbfgs_inverse_hessian(H₀::Diagonal, S0, Y0, history_ind, history_length)
    J = history_length
    α = H₀.diag
    B = similar(α, size(α, 1), 2J)
    D = fill!(similar(α, 2J, 2J), false)
    iszero(J) && return WoodburyPDMat(H₀, B, D)

    hist_inds = [(history_ind + 1):history_length; 1:history_ind]
    @views begin
        S = S0[:, hist_inds]
        Y = Y0[:, hist_inds]
        B₁ = B[:, 1:J]
        B₂ = B[:, (J + 1):(2J)]
        D₁₁ = D[1:J, 1:J]
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
    tril!(D₂₂) # eliminate all but diagonal
    mul!(D₂₂, Y', B₁, true, true)
    LinearAlgebra.copytri!(D₂₂, 'U', false, false)
    rmul!(D₂₂, nRinv)
    lmul!(nRinv′, D₂₂)

    return WoodburyPDMat(H₀, B, D)
end
