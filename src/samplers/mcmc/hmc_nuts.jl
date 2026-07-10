# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Native multinomial NUTS (no-U-turn sampler) core for an identity-metric
# Hamiltonian. BAT tunes parameter-space transformations instead of mass
# matrices, so the kinetic energy is always |p|²/2 here.
#
# References:
#
# * Hoffman & Gelman (2014), "The No-U-Turn Sampler: Adaptively Setting Path
#   Lengths in Hamiltonian Monte Carlo"
# * Betancourt (2017), "A Conceptual Introduction to Hamiltonian Monte Carlo"
#   (multinomial sampling and the generalized no-U-turn criterion, sec. A.4)
# * Stan reference implementation (biased progressive sampling and the
#   additional U-turn checks between merged subtrees,
#   https://github.com/stan-dev/stan/pull/2800)


# Position q with momentum p, target log-density logd and its gradient at q,
# and total energy H = |p|²/2 - logd. Non-finite phase points get H = Inf so
# that trajectory termination treats them as divergent.
struct HMCPhasePoint{T<:Real,V<:AbstractVector{<:Real}}
    q::V
    p::V
    logd::T
    grad::V
    H::T
end

function _hmc_phasepoint(q::AbstractVector{<:Real}, p::AbstractVector{<:Real}, logd::Real, grad::AbstractVector{<:Real})
    H = sum(abs2, p) / 2 - logd
    if isfinite(H) && all(isfinite, grad)
        HMCPhasePoint(q, p, logd, grad, H)
    else
        HMCPhasePoint(q, p, oftype(logd, -Inf), grad, oftype(H, Inf))
    end
end

function _hmc_phasepoint(f_logdgrad::Function, q::AbstractVector{<:Real}, p::AbstractVector{<:Real})
    logd, grad = f_logdgrad(q)
    _hmc_phasepoint(q, p, logd, grad)
end


function _leapfrog_step(f_logdgrad::Function, z::HMCPhasePoint, stepsize::Real)
    p_half = z.p .+ stepsize / 2 .* z.grad
    q_new = z.q .+ stepsize .* p_half
    logd_new, grad_new = f_logdgrad(q_new)
    p_new = p_half .+ stepsize / 2 .* grad_new
    return _hmc_phasepoint(q_new, p_new, logd_new, grad_new)
end


# Binary trajectory tree, stores only its edges, the multinomially sampled
# candidate, the sum of leaf momenta (generalized-no-U-turn statistic) and
# summary statistics. logw is the log total multinomial weight of the tree,
# relative to the initial energy H0.
struct HMCBinaryTree{P<:HMCPhasePoint,V<:AbstractVector{<:Real},T<:Real}
    zleft::P
    zright::P
    zcand::P
    psum::V
    logw::T
    sum_alpha::T
    nsteps::Int
end

function _hmc_tree_leaf(z::HMCPhasePoint, H0::Real, max_delta_energy::Real)
    dH = z.H - H0
    alpha = exp(min(zero(dH), -dH))
    divergent = !(dH < max_delta_energy)
    return HMCBinaryTree(z, z, z, z.p, -dH, alpha, 1), divergent
end

function _logaddexp(a::Real, b::Real)
    m = max(a, b)
    isfinite(m) ? m + log1p(exp(-abs(a - b))) : m
end

function _merge_trees(tl::HMCBinaryTree, tr::HMCBinaryTree, zcand::HMCPhasePoint, logw::Real)
    HMCBinaryTree(
        tl.zleft, tr.zright, zcand,
        tl.psum + tr.psum, logw,
        tl.sum_alpha + tr.sum_alpha,
        tl.nsteps + tr.nsteps
    )
end

# Merge two adjacent subtrees, sampling the candidate multinomially, i.e.
# proportionally to the subtree weights.
function _combine_subtrees(rng::AbstractRNG, tl::HMCBinaryTree, tr::HMCBinaryTree)
    logw = _logaddexp(tl.logw, tr.logw)
    zcand = logw < tl.logw + randexp(rng, typeof(logw)) ? tl.zcand : tr.zcand
    return _merge_trees(tl, tr, zcand, logw)
end

_is_uturn(psum::AbstractVector{<:Real}, pleft::AbstractVector{<:Real}, pright::AbstractVector{<:Real}) =
    dot(psum, pleft) <= 0 || dot(psum, pright) <= 0

# Generalized no-U-turn condition for the merged tree, including the
# additional checks at the subtree boundaries used by Stan.
function _tree_turning(tl::HMCBinaryTree, tr::HMCBinaryTree, t::HMCBinaryTree)
    _is_uturn(t.psum, t.zleft.p, t.zright.p) ||
        _is_uturn(tl.psum + tr.zleft.p, t.zleft.p, tr.zleft.p) ||
        _is_uturn(tl.zright.p + tr.psum, tl.zright.p, t.zright.p)
end

# Recursively build a trajectory tree of the given depth, starting with one
# leapfrog step from z_edge in direction dir. Returns (tree, divergent,
# turning); the returned tree is incomplete if divergent or turning is true.
function _build_hmc_tree(
    rng::AbstractRNG, f_logdgrad::Function, z_edge::HMCPhasePoint,
    dir::Int, depth::Int, stepsize::Real, H0::Real, max_delta_energy::Real
)
    if depth == 0
        z_new = _leapfrog_step(f_logdgrad, z_edge, dir * stepsize)
        tree, divergent = _hmc_tree_leaf(z_new, H0, max_delta_energy)
        return tree, divergent, false
    else
        tree1, divergent, turning = _build_hmc_tree(rng, f_logdgrad, z_edge, dir, depth - 1, stepsize, H0, max_delta_energy)
        if divergent || turning
            return tree1, divergent, turning
        end
        z_next = dir > 0 ? tree1.zright : tree1.zleft
        tree2, divergent, turning = _build_hmc_tree(rng, f_logdgrad, z_next, dir, depth - 1, stepsize, H0, max_delta_energy)
        tl, tr = dir > 0 ? (tree1, tree2) : (tree2, tree1)
        tree = _combine_subtrees(rng, tl, tr)
        return tree, divergent, turning || _tree_turning(tl, tr, tree)
    end
end


"""
    hmc_nuts_transition(
        rng::AbstractRNG, f_logdgrad::Function, z0::HMCPhasePoint,
        stepsize::Real, max_depth::Integer, max_delta_energy::Real
    )

*BAT-internal, not part of stable public API.*

Perform a single multinomial NUTS transition from phase point `z0`, doubling
the trajectory until the generalized no-U-turn condition triggers, the energy
error exceeds `max_delta_energy` or the tree depth reaches `max_depth`.

`f_logdgrad(q)` must return the tuple `(logd, grad)` of the target
log-density and its gradient.

Returns a `NamedTuple` with the fields `z` (the sampled phase point),
`p_accept` (average leapfrog Metropolis acceptance probability), `depth`,
`n_leapfrog` and `divergent`.
"""
function hmc_nuts_transition(
    rng::AbstractRNG, f_logdgrad::Function, z0::HMCPhasePoint,
    stepsize::Real, max_depth::Integer, max_delta_energy::Real
)
    H0 = z0.H
    logw0 = zero(H0)
    tree = HMCBinaryTree(z0, z0, z0, z0.p, logw0, zero(H0), 0)
    divergent = false
    depth = 0

    while depth < max_depth
        dir = rand(rng, Bool) ? 1 : -1
        z_edge = dir > 0 ? tree.zright : tree.zleft
        subtree, div_sub, turn_sub = _build_hmc_tree(rng, f_logdgrad, z_edge, dir, depth, stepsize, H0, max_delta_energy)
        divergent |= div_sub
        subtree_ok = !(div_sub || turn_sub)

        # Biased progressive sampling: prefer the new subtree's candidate
        # with probability min(1, w_subtree / w_tree).
        zcand = tree.zcand
        if subtree_ok
            depth += 1
            if tree.logw < subtree.logw + randexp(rng, typeof(tree.logw))
                zcand = subtree.zcand
            end
        end

        tl, tr = dir > 0 ? (tree, subtree) : (subtree, tree)
        tree = _merge_trees(tl, tr, zcand, _logaddexp(tl.logw, tr.logw))
        if !subtree_ok || _tree_turning(tl, tr, tree)
            break
        end
    end

    p_accept = tree.sum_alpha / max(tree.nsteps, 1)
    return (z = tree.zcand, p_accept = p_accept, depth = depth, n_leapfrog = tree.nsteps, divergent = divergent)
end


"""
    hmc_find_good_stepsize(
        rng::AbstractRNG, f_logdgrad::Function, q0::AbstractVector{<:Real};
        init_stepsize::Real = 0.1, max_niters::Integer = 100
    )

*BAT-internal, not part of stable public API.*

Heuristically search a leapfrog step size at position `q0` for which the
single-step Metropolis acceptance ratio lies between 1/4 and 3/4 (see
Hoffman & Gelman (2014), algorithm 4).
"""
function hmc_find_good_stepsize(
    rng::AbstractRNG, f_logdgrad::Function, q0::AbstractVector{<:Real};
    init_stepsize::Real = 0.1, max_niters::Integer = 100
)
    T = float(eltype(q0))
    z0 = _hmc_phasepoint(f_logdgrad, q0, randn(rng, T, length(q0)))
    log_accept_ratio(stepsize) = z0.H - _leapfrog_step(f_logdgrad, z0, stepsize).H
    log_a_min, log_a_cross, log_a_max = 2 * log(T(1//2)), log(T(1//2)), log(T(3//4))

    # Double/halve until the acceptance ratio crosses 1/2:
    stepsize = T(init_stepsize)
    ratio_too_high = log_accept_ratio(stepsize) > log_a_cross
    stepsize_lo = stepsize_hi = stepsize
    for _ in 1:max_niters
        stepsize_new = ratio_too_high ? 2 * stepsize : stepsize / 2
        if xor(ratio_too_high, log_accept_ratio(stepsize_new) > log_a_cross)
            stepsize_lo, stepsize_hi = minmax(stepsize, stepsize_new)
            break
        end
        stepsize = stepsize_new
        stepsize_lo = stepsize_hi = stepsize
    end

    # Bisect until the acceptance ratio lies in [1/4, 3/4]:
    for _ in 1:max_niters
        stepsize_mid = (stepsize_lo + stepsize_hi) / 2
        la = log_accept_ratio(stepsize_mid)
        if la > log_a_max
            stepsize_lo = stepsize_mid
        elseif la < log_a_min
            stepsize_hi = stepsize_mid
        else
            return stepsize_mid
        end
    end
    return stepsize_lo
end
