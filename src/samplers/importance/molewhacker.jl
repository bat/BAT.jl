# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    MolewhackerRefit(; kwargs...)

Importance-weighted EM refit of the [`MolewhackerSampling`](@ref) proposal to its
fresh-round draws, in the style of MitISEM (Hoogerheide, Opschoor and van Dijk).
The number of fitted Gaussians maximizes the weighted log-likelihood of held-out draws.

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MolewhackerRefit
    "Largest number of fitted Gaussians."
    maxcomponents::Int = 6
    "Mass share that the adaptive mixture keeps for defence."
    defence::Float64 = 0.2
    "Covariance shrinkage toward the diagonal."
    shrinkage::Float64 = 0.1
    "Covariance inflation of the fitted Gaussians. Wider fits bound the weights where the target tails are Gaussian."
    inflation::Float64 = 1.1
    "Effective draws needed per dimension and fitted Gaussian."
    min_ess_per_dim::Float64 = 2.0
    "Fitted mass below which a Gaussian drops out."
    min_mass::Float64 = 1e-3
    "Maximum EM iterations per fit."
    maxiter::Int = 200
    "Stop EM when the weighted log-likelihood per draw rises by less than this."
    tol::Float64 = 1e-6
end
export MolewhackerRefit

"""
    MolewhackerSampling(; kwargs...)

Adaptive Gaussian-mixture importance sampling with local Fisher geometry.
Finds initial modes, adds Gaussians at high target-to-proposal ratios, and
updates all component masses from center ratios. Requires a differentiable
forward-model likelihood and a standard-normal prior after `pretransform`.

The adaptive pool guides proposal construction only. Final samples are fresh
IID draws from the frozen mixture, with optional prior mixing.
Weights are `exp(logtarget - logproposal - logweight_scale)`; the common
scale is retained in `evalinfo.result`.

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MolewhackerSampling{TR<:TransformIntent,IA,IM,E<:BATExecutor,D,RF<:Union{Nothing,MolewhackerRefit}} <: AbstractSamplingAlgorithm
    pretransform::TR = NormalBased()
    "Seed source in original coordinates, or `nothing` for Sobol starts in normal coordinates."
    init::IA = nothing
    "Number of initial seeds. Zero skips mode initialization."
    nseeds::Int = 10
    "Add a Laplace Gaussian (observed information) beside each seed's Fisher Gaussian, and Newton-polish the seed."
    laplace_seeds::Bool = true
    "Laplace variance inflation. On the public DeepCore model, 1.2 beat both 1.0 and 1.5."
    laplace_inflation::Float64 = 1.2
    "Seed optimizer. The default L-BFGS-B backend requires `import OptimizationLBFGSB`. It stops after 50 iterations with Laplace seeds, whose Newton step finishes the search."
    init_mode::IM = nseeds == 0 ? nothing : OptimizationAlg(optalg = ext_default(pkgext(Val(:OptimizationLBFGSB)), Val(:LBFGSB_ALG)), maxiters = laplace_seeds ? 50 : 1_000)
    "Production count, or its cap with finite `target_ess`. Nothing follows the pool size or ESS goal."
    nsamples::Union{Nothing,Int} = nothing
    "Optional production ESS goal. A fresh pilot sizes production; achieved ESS is not guaranteed."
    target_ess::Float64 = Inf
    "Stop adaptation when recycled-pool ESS exceeds this heuristic threshold."
    target_pool_ess::Float64 = Inf
    "Stop adaptation when recycled-pool ESS per point exceeds this heuristic threshold."
    target_efficiency::Float64 = Inf
    "Initial discovery pool and production-sizing pilot count."
    batchsize::Int = 1000
    maxiter::Int = 100
    "Maximum component proposals, including repeated selections and any added prior component."
    maxcomponents::Int = typemax(Int)
    "Maximum target calls, excluding geometry. No additional call limit by default."
    maxevals::Int = typemax(Int)
    "Number of candidate centers added per round, independent of the thread count."
    ncandidates::Int = 14
    "Optional prior coefficient added after fitting. Zero leaves the fitted proposal unchanged."
    exploration_mass::Float64 = 0.0
    executor::E = default_executor()
    "Optional function of final log importance ratios."
    weight_diagnostic::D = nothing
    "Pareto-smooth the largest production weights (PSIS). Lowers variance and adds a small bias."
    smooth_weights::Bool = false
    "Rounds after adaptation stops that first add `batchsize` fresh proposal draws to the pool."
    fresh_rounds::Int = 3
    "Refit of the proposal to the fresh-round draws, or `nothing` to keep the adaptive mixture."
    refit::RF = MolewhackerRefit()
end
export MolewhackerSampling

function _mw_check(alg, context)
    @argcheck (isnothing(alg.nsamples) || alg.nsamples > 0) && alg.batchsize > 0
    @argcheck alg.maxiter >= 0 && alg.nseeds >= 0 && alg.ncandidates > 0 && alg.fresh_rounds >= 0
    # Laplace seeds can double the initial occurrences.
    @argcheck alg.maxcomponents >= max(1, (1 + alg.laplace_seeds) * alg.nseeds + Int(alg.exploration_mass > 0))
    @argcheck alg.target_ess > 0 && 0 <= alg.exploration_mass < 1
    @argcheck alg.target_pool_ess > 0 && alg.target_efficiency > 0
    @argcheck alg.target_efficiency <= 1 || alg.target_efficiency == Inf
    pilot_count = isfinite(alg.target_ess) ? alg.batchsize : 0
    production_count = something(alg.nsamples, alg.batchsize)
    @argcheck alg.maxevals >= production_count && alg.maxevals - production_count >= pilot_count
    reserve = production_count + pilot_count
    discovery_count = alg.maxiter > 0 ? alg.batchsize : 0
    center_count = alg.nseeds > 0 ? alg.nseeds : Int(alg.maxiter > 0)
    @argcheck alg.maxevals - reserve >= discovery_count + center_count "Leave target calls for initial centers and discovery, or disable initialization and adaptation."
    @argcheck isnothing(alg.init_mode) || alg.maxevals - reserve - discovery_count - center_count >= alg.nseeds "Leave target calls for mode searches, or disable the seed optimizer."
    @argcheck get_compute_unit(context) isa CPUnit
    @argcheck iszero(alg.exploration_mass) || 0 < get_precision(context)(alg.exploration_mass) < 1
    @argcheck alg.laplace_inflation > 0
    refit = alg.refit
    @argcheck isnothing(refit) || (refit.maxcomponents > 0 && 0 <= refit.defence < 1 && 0 <= refit.shrinkage <= 1 &&
        refit.inflation > 0 && refit.min_ess_per_dim > 0 && 0 <= refit.min_mass < 1 && refit.maxiter > 0 && refit.tol >= 0)
    return reserve
end

function _mw_gaussian(center::AbstractVector{T}, precision) where T
    P = PDMat{T}(precision)
    c = copy(center)
    return MvNormalCanon(c, P * c, P)
end

# Idle tasks split Jacobian columns when fewer geometries than tasks are pending.
# Blocks keep at least three columns: narrower blocks added allocation but no speed.
_mw_jacobian_blocks(executor::MultiThreadedExec, n, dim) = clamp(fld(executor.ntasks, max(n, 1)), 1, cld(dim, 3))
_mw_jacobian_blocks(::BATExecutor, n, dim) = 1

function _mw_draw(q, logtarget, n, executor, context)
    v = VectorOfSimilarVectors(rand(get_rng(context), q, n))
    first_logp = logtarget(first(v))
    logp = Vector{typeof(float(first_logp))}(undef, n)
    logp[1] = first_logp
    exec_map!(logtarget, executor, view(logp, 2:n), view(v, 2:n))
    all(x -> isfinite(x) || x == -Inf, logp) || throw(ArgumentError("MolewhackerSampling encountered an invalid target log density."))
    logr = _mw_batched_logpdf(q, flatview(v), executor)
    all(isfinite, logr) || throw(ArgumentError("MolewhackerSampling encountered a non-finite generating log density."))
    return (; v, logp, logr)
end

function _mw_batched_logpdf(d, x::AbstractMatrix, executor = default_executor())
    T = promote_type(Distributions.partype(d), eltype(x))
    # Mixture logpdf! stores component densities in the mixture-weight type.
    if d isa MixtureModel && T != eltype(probs(d))
        return logpdf.(Ref(d), eachcol(x))
    end
    r = Vector{T}(undef, size(x, 2))
    logpdf!(r, d, x)
    if d isa MixtureModel
        # The batch kernel can produce NaN when every component returns -Inf.
        for i in eachindex(r)
            isnan(r[i]) && (r[i] = logpdf(d, view(x, :, i)))
        end
    end
    return r
end

function _mw_batched_logpdf(d::MixtureModel{Multivariate,Continuous,<:MvNormalCanon}, x::AbstractMatrix,
    executor = default_executor())
    T = promote_type(eltype(mean(first(d.components))), eltype(x), eltype(probs(d)))
    n = size(x, 2)
    n == 0 && return T[]
    ntasks = executor isa MultiThreadedExec ? executor.ntasks : Threads.nthreads()
    nchunks = executor isa SequentialExec ? 1 :
        min(ntasks, n, max(1, n * length(d.components) ÷ 65536))
    logweights = log.(probs(d))
    constants = Distributions.mvnormal_c0.(d.components)
    nchunks == 1 && return _mw_gaussian_logpdf(d, x, logweights, constants)
    # Keep matrix batches wide when the pool is small relative to the worker count.
    by_component = n < 256nchunks
    count = by_component ? length(d.components) : n
    width = cld(count, nchunks)
    ranges = [i:min(i + width - 1, count) for i in 1:width:count]
    blocks = Vector{Vector{T}}(undef, length(ranges))
    score = r -> by_component ? _mw_component_logpdf(d, x, constants, r) :
        _mw_gaussian_logpdf(d, view(x, :, r), logweights, constants)
    exec_map!(score, executor, blocks, ranges)
    if by_component
        result = first(blocks)
        for block in Iterators.drop(blocks, 1)
            result .= _logaddexp.(result, block)
        end
        return result
    end
    return reduce(vcat, blocks)
end

function _mw_component_logpdf(d, x, constants, indices)
    weights = probs(d)[indices]
    mass = sum(weights)
    T = promote_type(eltype(mean(first(d.components))), eltype(x), eltype(weights))
    iszero(mass) && return fill(T(-Inf), size(x, 2))
    part = MixtureModel(d.components[indices], weights ./ mass)
    result = _mw_gaussian_logpdf(part, x, log.(probs(part)), constants[indices])
    result .+= log(mass)
    return result
end

function _mw_gaussian_logpdf(d, x, logweights, constants)
    T = promote_type(eltype(mean(first(d.components))), eltype(x), eltype(probs(d)))
    n, k = size(x, 2), length(d.components)
    # Bound the density workspace to about 2 MiB per task in Float64, and reuse it.
    width = min(n, 512, max(1, 262144 ÷ k))
    terms = Matrix{T}(undef, width, k)
    shifted = Matrix{T}(undef, size(x, 1), width)
    maxima = Vector{T}(undef, width)
    result = zeros(T, n)
    for first in 1:width:n
        indices = first:min(first + width - 1, n)
        count = length(indices)
        delta = view(shifted, :, 1:count)
        fill!(maxima, T(-Inf))
        for i in eachindex(d.components)
            iszero(probs(d)[i]) && continue
            component = d.components[i]
            delta .= view(x, :, indices) .- mean(component)
            lmul!(cholesky(component.J).U, delta)
            for j in 1:count
                value = constants[i] - sum(abs2, view(delta, :, j)) / 2 + logweights[i]
                terms[j, i] = value
                maxima[j] = max(maxima[j], value)
            end
        end
        r = view(result, indices)
        for i in eachindex(d.components)
            iszero(probs(d)[i]) && continue
            for j in 1:count
                r[j] += exp(terms[j, i] - maxima[j])
            end
        end
        for j in 1:count
            r[j] = log(r[j]) + maxima[j]
            isnan(r[j]) && (r[j] = logpdf(d, view(x, :, indices[j])))
        end
    end
    return result
end

# Pool entries of the per-component log-density cache. Above this, scoring recomputes
# all component densities each round.
const _MW_SCORE_CACHE_LIMIT = 2^24

function _mw_logpdf_column(component, x)
    delta = x .- mean(component)
    lmul!(cholesky(component.J).U, delta)
    return Distributions.mvnormal_c0(component) .- vec(sum(abs2, delta, dims = 1)) ./ 2
end

# Rounds change every mass but no existing component density, so only new pool
# points and new components need whitening.
function _mw_extend_scores(cache, components, points, executor)
    n, k = size(cache)
    N, K = size(points, 2), length(components)
    grown = similar(cache, N, K)
    grown[1:n, 1:k] = cache
    columns = Vector{Vector{eltype(cache)}}(undef, K)
    rows = view(points, :, n+1:N)
    exec_map!(j -> _mw_logpdf_column(components[j], j <= k ? rows : points), executor, columns, collect(1:K))
    for j in 1:K
        j <= k ? (grown[n+1:N, j] = columns[j]) : (grown[:, j] = columns[j])
    end
    return grown
end

function _mw_cached_logpdf_rows(cache, logmass, r)
    T = eltype(cache)
    maxima = fill(T(-Inf), length(r))
    @inbounds for k in eachindex(logmass)
        lm, column = logmass[k], view(cache, r, k)
        @simd for j in eachindex(maxima)
            maxima[j] = max(maxima[j], column[j] + lm)
        end
    end
    sums = zeros(T, length(r))
    @inbounds for k in eachindex(logmass)
        lm, column = logmass[k], view(cache, r, k)
        isfinite(lm) || continue
        @simd for j in eachindex(sums)
            sums[j] += exp(column[j] + lm - maxima[j])
        end
    end
    return log.(sums) .+ maxima
end

function _mw_pool_logpdf(q, components, points, cache, executor, limit = _MW_SCORE_CACHE_LIMIT)
    if size(points, 2) * length(components) > limit
        return _mw_batched_logpdf(q, points, executor), similar(cache, 0, 0)
    end
    cache = _mw_extend_scores(cache, components, points, executor)
    n = size(cache, 1)
    ntasks = executor isa MultiThreadedExec ? executor.ntasks : 1
    width = cld(n, max(1, min(ntasks, n ÷ 256)))
    ranges = [i:min(i + width - 1, n) for i in 1:width:n]
    blocks = Vector{Vector{eltype(cache)}}(undef, length(ranges))
    logmass = log.(probs(q))
    exec_map!(r -> _mw_cached_logpdf_rows(cache, logmass, r), executor, blocks, ranges)
    return reduce(vcat, blocks), cache
end

struct MolewhackerBudgetReached <: Exception end

function _mw_with_budget(f, logtarget, remaining)
    ncalls = Ref(0)
    counted = x -> begin
        ChainRulesCore.ignore_derivatives() do
            ncalls[] < remaining || throw(MolewhackerBudgetReached())
            ncalls[] += 1
        end
        logtarget(x)
    end
    try
        return f(counted), ncalls[], false
    catch err
        err isa MolewhackerBudgetReached || rethrow()
        return nothing, ncalls[], true
    end
end

function _mw_mode(center, logtarget, mode, remaining, context)
    isnothing(mode) && return center, 0, false
    result, ncalls, exhausted = _mw_with_budget(logtarget, remaining) do counted
        # Optimizer results do not infer. The assertion keeps later seed work concrete.
        convert(typeof(center), maximize_density(counted, center, mode, context).result)::typeof(center)
    end
    return exhausted ? center : result, ncalls, exhausted
end

# Mixture refinement (MolewhackerRefit): fit Gaussians to the target by importance-weighted EM on
# the fresh-round draws, each weighted by its own proposal, and keep the adaptive mixture as a
# defensive share. Center ratios cannot see proposal mass placed where the target is small. These
# draws can. The held-out weighted log-likelihood is the cross-entropy part of KL(p || fit).
function _mw_weighted_moments(x, w, shrinkage)
    mu = x * w ./ sum(w)
    c = x .- mu
    S = (c .* w') * c' ./ sum(w)
    # Shrink toward the diagonal: a few effective draws must fit many covariances.
    return mu, Symmetric((1 - shrinkage) .* S .+ shrinkage .* Diagonal(diag(S)))
end

# M step: weighted moments per component. Components with negligible mass drop out.
function _mw_em_masses(x, w, R, refit)
    T = eltype(x)
    mass = vec(sum(w .* R, dims = 1))
    keep = findall(>(T(refit.min_mass)), mass)
    return [_mw_weighted_moments(x, w .* view(R, :, k), T(refit.shrinkage)) for k in keep], mass[keep] ./ sum(mass[keep])
end

# Gaussian log density from the Cholesky factor. MvNormal with a Symmetric covariance does not infer.
function _mw_normal_logpdf(mu, S, x)
    U = cholesky(S).U
    z = U' \ (x .- mu)
    return .-vec(sum(abs2, z, dims = 1)) ./ 2 .- (logdet(U) + size(x, 1) * log(2 * eltype(x)(pi)) / 2)
end

# E step: responsibilities of each component for each draw, and each draw's mixture log density.
function _mw_em_estep(x, fits, pis)
    L = reduce(hcat, [log(p) .+ _mw_normal_logpdf(mu, S, x) for ((mu, S), p) in zip(fits, pis)])
    m = maximum(L, dims = 2)
    R = exp.(L .- m)
    s = sum(R, dims = 2)
    return R ./ s, vec(m .+ log.(s))
end

# Weighted k-means++ starts, with hard assignment to the nearest start.
function _mw_em_start(x, w, K, rng)
    T = eltype(x)
    centers = [x[:, argmax(w)]]
    while length(centers) < K
        d2 = [minimum(c -> sum(abs2, view(x, :, i) .- c), centers) for i in axes(x, 2)] .* w
        push!(centers, x[:, something(findfirst(>=(rand(rng, T) * sum(d2)), cumsum(d2)), size(x, 2))])
    end
    R = zeros(T, size(x, 2), K)
    for i in axes(x, 2)
        R[i, argmin(k -> sum(abs2, view(x, :, i) .- centers[k]), 1:K)] = 1
    end
    return R
end

function _mw_em(x, w, R, refit)
    T = eltype(x)
    fits, pis = Tuple{Vector{T},Symmetric{T,Matrix{T}}}[], T[]
    loglik = T(-Inf)
    for _ in 1:refit.maxiter
        fits, pis = _mw_em_masses(x, w, R, refit)
        all(f -> isposdef(last(f)), fits) || return nothing
        R, l = _mw_em_estep(x, fits, pis)
        previous, loglik = loglik, dot(w, l)
        loglik - previous < refit.tol && break
    end
    return fits, pis
end

function _mw_fit_mixture(q, x, logw, refit, context)
    T = eltype(x)
    w = exp.(logw .- maximum(logw))
    fit, held = x[:, 1:2:end], x[:, 2:2:end]
    wfit, wheld = w[1:2:end] ./ sum(w[1:2:end]), w[2:2:end] ./ sum(w[2:2:end])
    best, score = nothing, T(-Inf)
    for K in 1:refit.maxcomponents
        1 / sum(abs2, wfit) > refit.min_ess_per_dim * K * size(x, 1) || break
        em = _mw_em(fit, wfit, _mw_em_start(fit, wfit, K, get_rng(context)), refit)
        isnothing(em) && continue
        s = dot(wheld, last(_mw_em_estep(held, em...)))
        s > score && ((best, score) = (em, s))
    end
    isnothing(best) && return q
    # Refit on all draws from the held-out winner. A new start can fall into a worse optimum.
    w ./= sum(w)
    em = _mw_em(x, w, first(_mw_em_estep(x, best...)), refit)
    isnothing(em) && return q
    fits, pis = em
    fitted = [_mw_gaussian(mu, Matrix(Symmetric(inv(cholesky(S)))) ./ T(refit.inflation)) for (mu, S) in fits]
    defence = T(refit.defence)
    masses = vcat(defence .* probs(q), (1 - defence) .* pis)
    return MixtureModel(vcat(q.components, fitted), masses ./ sum(masses))
end

function _mw_efficiency(logw)
    c = maximum(logw)
    isfinite(c) || throw(ArgumentError("MolewhackerSampling drew no finite positive target mass."))
    w = exp.(logw .- c)
    ess = sum(w)^2 / sum(abs2, w)
    return (; weight = w, ess, efficiency = ess / length(w), logweight_scale = c)
end

# Generalized Pareto fit to the largest importance ratios, relative to the ratio at the tail
# cutoff (Zhang & Stephens 2009). The shape gets the weakly informative adjustment of PSIS.
function _mw_pareto_fit(logw)
    finite = findall(isfinite, logw)
    order = finite[sortperm(logw[finite])]
    n = length(order)
    ntail = min(ceil(Int, n / 5), ceil(Int, 3sqrt(n)))
    n > ntail >= 5 || return nothing
    tail = order[n-ntail+1:n]
    # Shift by the largest log ratio so exceedances stay in [0, 1] for any weight spread.
    logmax = float(logw[last(tail)])
    base = exp(logw[order[n-ntail]] - logmax)
    x = exp.(logw[tail] .- logmax) .- base
    last(x) > 0 || return nothing
    m = 30 + floor(Int, sqrt(ntail))
    xstar = x[max(1, floor(Int, ntail / 4 + 1 / 2))]
    # Tail spread beyond the Float64 range: the largest ratios dominate completely.
    xstar > 0 || return (; k = oftype(logmax, Inf), sigma = oftype(logmax, NaN), tail, logmax, base)
    θ = [1 / last(x) + (1 - sqrt(m / (j - 1 / 2))) / (3xstar) for j in 1:m]
    ks = [mean(log1p.(-t .* x)) for t in θ]
    loglik = ntail .* (log.(-θ ./ ks) .- ks .- 1)
    weights = exp.(loglik .- maximum(loglik))
    θhat = sum(θ .* weights) / sum(weights)
    k = mean(log1p.(-θhat .* x))
    return (; k = (ntail * k + 5) / (ntail + 10), sigma = -k / θhat, tail, logmax, base)
end

# ESS misses a single dominant weight; the tail shape does not.
_mw_pareto_k(logw) = (fit = _mw_pareto_fit(logw); isnothing(fit) ? NaN : fit.k)

# PSIS: replace the largest ratios by expected GPD order statistics, capped at the largest raw ratio.
function _mw_smooth(logw)
    fit = _mw_pareto_fit(logw)
    (isnothing(fit) || !isfinite(fit.k)) && return logw
    p =((1:length(fit.tail)) .- 1 / 2) ./ length(fit.tail)
    q = abs(fit.k) < sqrt(eps()) ? -fit.sigma .* log1p.(-p) : fit.sigma .* expm1.(-fit.k .* log1p.(-p)) ./ fit.k
    smoothed = copy(logw)
    smoothed[fit.tail] .= fit.logmax .+ min.(log.(fit.base .+ q), 0)
    return smoothed
end

# BAT's weightedmeasure represents a likelihood scale as a log-function chain.
# Recover only these known constant shifts, never an arbitrary log-density model.
_mw_model(likelihood::Union{Likelihood,_SimpleLikelihood}) = _get_model(likelihood)
_mw_model(likelihood::DensityInterface.LogFuncDensity) = _mw_logmodel(likelihood._log_f)
_mw_logmodel(f::Base.Fix1{typeof(logdensityof)}) = _mw_model(f.x)
function _mw_logmodel(f::FunctionChain)
    fs = fchainfs(f)
    tail = last(fs)
    if tail isa Base.Fix2{typeof(+),<:Real} && isfinite(tail.x)
        return _mw_logmodel(ffchain(Base.front(fs)...))
    end
    throw(ArgumentError("MolewhackerSampling requires a supported forward-model likelihood."))
end
_mw_logmodel(f) = throw(ArgumentError("MolewhackerSampling requires a supported forward-model likelihood."))


function _mw_center_mixture(components, center_logp, executor, previous = similar(center_logp, 0),
    multiplicities = ones(Int, length(components)), added = length(previous)+1:length(components))
    T = eltype(first(components))
    n, nold = sum(multiplicities), length(previous)
    uniform = MixtureModel(components, T.(multiplicities) ./ T(n))
    centers = reduce(hcat, mean.(components))
    # Existing center/component pairs never change. Add only the new densities
    # to their unnormalized sums, retaining every component's multiplicity.
    new_uniform = MixtureModel(components[added], fill(inv(T(length(added))), length(added)))
    old_terms = _mw_batched_logpdf(new_uniform, view(centers, :, 1:nold), executor) .+ log(T(length(added)))
    new_terms = _mw_batched_logpdf(uniform, view(centers, :, nold+1:length(components)), executor) .+ log(T(n))
    center_logsum = vcat(_logaddexp.(previous, old_terms), new_terms)
    logw = center_logp .- (center_logsum .- log(T(n)))
    offset = maximum(logw)
    isfinite(offset) || return nothing, center_logsum
    weights = T.(exp.(logw .- offset)) .* multiplicities
    return MixtureModel(components, weights ./ sum(weights)), center_logsum
end

function _mw_initial_proposal(transformed_m, f, model, logtarget, gprior, alg, fit_budget, ad, context)
    T = get_precision(context)
    components = typeof(gprior)[]
    nevals, ngeometries, nfailed, nexhausted, nhessians = 0, 0, 0, 0, 0
    if alg.nseeds > 0
        seeds = if isnothing(alg.init)
            bat_sample(StandardMvNormal{T}(length(gprior)), SobolSampler(nsamples = alg.nseeds), context).result.v
        else
            bat_initval(transformed_m, alg.nseeds, apply_trafo_to_init(f, alg.init), context).result
        end
        starts = [T.(collect(seed)) for seed in seeds]
        results = Vector{Tuple{Vector{T},Int,Bool}}(undef, alg.nseeds)
        if isnothing(alg.init_mode)
            results .= [(c, 0, false) for c in starts]
        else
            # Preserve center-density and discovery calls after the parallel searches.
            mode_budget = fit_budget - alg.nseeds - (alg.maxiter > 0 ? alg.batchsize : 0)
            share, extra = divrem(mode_budget, alg.nseeds)
            contexts = [set_rng(context, Philox4x((rand(get_rng(context), UInt64), UInt64(i)))::Philox4x{UInt64,10}) for i in 1:alg.nseeds]
            search = i -> _mw_mode(starts[i], logtarget, deepcopy(alg.init_mode), share + (i <= extra), contexts[i])
            exec_map!(search, alg.executor, results, collect(1:alg.nseeds))
        end
        nevals = sum(r -> r[2], results)
        nexhausted = count(r -> r[3], results)
        centers = [r[1] for r in results if !r[3]]
        precisions = Vector{Union{Nothing,typeof(gprior.J)}}(undef, length(centers))
        nblocks = _mw_jacobian_blocks(alg.executor, length(centers), length(gprior))
        exec_map!(c -> _mw_local_precision(model, c, ad, nblocks), alg.executor, precisions, centers)
        ngeometries = length(centers)
        keep = findall(!isnothing, precisions)
        nfailed = ngeometries - length(keep)
        seeds, fisher = centers[keep], precisions[keep]
        observed = Vector{Union{Nothing,Matrix{T}}}(nothing, length(seeds))
        if alg.laplace_seeds && !isempty(seeds)
            seeds, observed, nhessians, ncalls = _mw_laplace_seeds(logtarget, seeds, fisher, ad, alg.executor)
            nevals += ncalls
        end
        components = typeof(gprior)[_mw_gaussian(seeds[i], fisher[i]) for i in eachindex(seeds)]
        # A Laplace Gaussian shares its seed's center, so center-ratio fitting gives the pair equal mass.
        for i in eachindex(seeds)
            isnothing(observed[i]) || push!(components, _mw_gaussian(seeds[i], observed[i] ./ T(alg.laplace_inflation)))
        end
    end
    center_logp = logtarget.(mean.(components))
    nevals += length(components)
    q, center_logsum = isempty(components) ? (nothing, similar(center_logp, 0)) :
        _mw_center_mixture(components, center_logp, alg.executor)
    if isnothing(q)
        q = MixtureModel([gprior], [one(T)])
        center_logp = alg.maxiter > 0 ? [logtarget(mean(gprior))] : Float64[]
        center_logsum = alg.maxiter > 0 ? [logpdf(gprior, mean(gprior))] : Float64[]
        nevals += length(center_logp)
    end
    return q, center_logp, center_logsum, nevals, ngeometries, nfailed, nexhausted, nhessians
end

# Laplace seeds: the observed information −∇²log p at each seed, from central differences of AD
# gradients run on the executor. Fisher information misses curvature where the forward model is
# stationary. Where the observed information is positive definite, one Newton step polishes the
# center if it raises the log target. Seeds within squared Mahalanobis 1 of an earlier seed share
# its Hessian: typical draws of a d-dimensional Gaussian lie near d, so this holds in any dimension.
function _mw_laplace_seeds(logtarget, centers, fisher, ad, executor)
    T = eltype(first(centers))
    d = length(first(centers))
    owner = collect(eachindex(centers))
    for j in eachindex(centers)
        i = findfirst(i -> owner[i] == i && dot(centers[j] - centers[i], fisher[i] * (centers[j] - centers[i])) < 1, 1:j-1)
        isnothing(i) || (owner[j] = i)
    end
    distinct = findall(j -> owner[j] == j, eachindex(centers))
    h = cbrt(eps(T))
    valgrad = valgrad_func(logtarget, ad)
    # Per distinct seed: the gradient at the center, then at ±h along each coordinate.
    offsets = [(0, zero(T)); [(k, σ * h) for k in 1:d for σ in (1, -1)]]
    jobs = [(s, k, δ) for s in distinct for (k, δ) in offsets]
    evaluated = Vector{Tuple{T,Vector{T}}}(undef, length(jobs))
    exec_map!(job -> _mw_shifted_valgrad(valgrad, centers[job[1]], job[2], job[3]), executor, evaluated, jobs)
    observed = Vector{Union{Nothing,Matrix{T}}}(nothing, length(centers))
    polished = copy(centers)
    ncalls = 0
    for (n, s) in enumerate(distinct)
        block = evaluated[(n-1)*length(offsets)+1:n*length(offsets)]
        H = reduce(hcat, [(last(block[2k]) .- last(block[2k+1])) ./ (2h) for k in 1:d])
        P = -Matrix(Symmetric((H + H') ./ 2))
        all(isfinite, P) && isposdef(Symmetric(P)) || continue
        for j in findall(==(s), owner)
            observed[j] = P
        end
        candidate = centers[s] .+ P \ last(first(block))
        ncalls += 1
        logtarget(candidate) > first(first(block)) && (polished[s] = candidate)
    end
    return polished, observed, length(distinct), ncalls
end

function _mw_shifted_valgrad(valgrad, center, k, δ)
    x = copy(center)
    k > 0 && (x[k] += δ)
    value, gradient = valgrad(x)
    return (value, collect(gradient))
end

function evalmeasure_impl(em::EvaluatedMeasure, alg::MolewhackerSampling, context::BATContext)
    reserve = _mw_check(alg, context)
    transformed_m, f = transform_and_unshape(alg.pretransform, em, context)
    m = unevaluated(transformed_m)
    is_std_mvnormal(getprior(m)) || throw(ArgumentError("MolewhackerSampling requires a standard-normal prior after pretransform."))
    model = ffcomp(_mw_model(getlikelihood(unevaluated(em))), inverse(f))
    logtarget = checked_logdensityof(m)
    T = get_precision(context)
    dim = some_dof(transformed_m)
    gprior = _mw_gaussian(zeros(T, dim), Matrix{T}(I, dim, dim))
    ad = alg.maxiter > 0 || alg.nseeds > 0 ? get_valid_adselector(context, alg) : get_adselector(context)
    fit_budget = alg.maxevals - reserve
    q, center_logp, center_logsum, nevals, ngeometries, nfailed, nseed_exhausted, nhessians =
        _mw_initial_proposal(transformed_m, f, model, logtarget, gprior, alg, fit_budget, ad, context)
    nseed_evals = nevals
    components = copy(q.components)
    multiplicities = ones(Int, length(components))
    ncomponent_proposals = length(components)
    has_prior = first(q.components) === gprior
    component_limit = alg.maxcomponents - Int(!has_prior && alg.exploration_mass > 0)
    # Automatic output reserves one fresh draw for each discovery point.
    draw_cost = isnothing(alg.nsamples) ? 2 : 1
    niterations, nfresh, npilot = 0, 0, 0
    stop_reason = nseed_exhausted > 0 && nseed_exhausted == alg.nseeds ? :maxevals : :maxiter
    history = StructArray((; iteration = Int[], fresh = Bool[], npilot = Int[], drawn = Int[], ncomponents = Int[], ncomponent_proposals = Int[]))
    pilot_ess = nothing
    # Pool and fresh draws grow in place across rounds.
    fresh_points, fresh_logw = ElasticArray{T}(undef, dim, 0), T[]

    if alg.maxiter > 0 && fit_budget - nevals >= alg.batchsize
        batch = _mw_draw(q, logtarget, alg.batchsize, alg.executor, context)
        nevals += alg.batchsize
        points, logp = ElasticArray{T}(flatview(batch.v)), batch.logp
        npilot = length(logp)
        # Zero marks unvisited points; -1 marks failed geometry. Positive entries
        # identify stored Gaussians, so reselection needs no equality search.
        component_index = zeros(Int, npilot)
        score_cache = Matrix{T}(undef, 0, 0)
        iteration, adapting = 0, true
        while true
            iteration += 1
            adapting &= iteration <= alg.maxiter
            if !adapting
                # The pool holds few draws from the current proposal, so it misses the ratio
                # spikes that production would hit. Fresh draws expose them to selection,
                # whatever stopped adaptation.
                nfresh < alg.fresh_rounds && fit_budget - nevals >= alg.batchsize * draw_cost &&
                    ncomponent_proposals < component_limit || break
                nfresh += 1
                batch = _mw_draw(q, logtarget, alg.batchsize, alg.executor, context)
                append!(fresh_points, flatview(batch.v))
                append!(fresh_logw, batch.logp .- batch.logr)
                nevals += alg.batchsize
                fit_budget -= (draw_cost - 1) * alg.batchsize
                append!(points, flatview(batch.v))
                append!(logp, batch.logp)
                append!(component_index, zeros(Int, alg.batchsize))
                npilot = length(logp)
            end
            logq, score_cache = _mw_pool_logpdf(q, components, points, score_cache, alg.executor)
            scores = logp .- logq
            reason = if !isfinite(maximum(scores))
                :no_finite_candidate
            elseif !adapting
                nothing
            # Recycled scores guide fitting only; they never become output weights.
            elseif (pilot_ess = _mw_efficiency(scores).ess) > alg.target_pool_ess
                :pilot_ess
            elseif pilot_ess / npilot > alg.target_efficiency
                :pool_efficiency
            elseif fit_budget - nevals < draw_cost
                :maxevals
            elseif ncomponent_proposals >= component_limit
                :maxcomponents
            end
            if !isnothing(reason)
                adapting || break
                stop_reason, adapting = reason, false
                continue
            end
            adapting && (niterations = iteration)
            nselected = min(alg.ncandidates, npilot, component_limit - ncomponent_proposals)
            indices = partialsortperm(scores, 1:nselected, rev = true)
            uncached = filter(i -> iszero(component_index[i]), indices)
            centers = collect.(eachcol(view(points, :, uncached)))
            precisions = Vector{Union{Nothing,typeof(gprior.J)}}(undef, length(uncached))
            nblocks = _mw_jacobian_blocks(alg.executor, length(uncached), dim)
            exec_map!(c -> _mw_local_precision(model, c, ad, nblocks), alg.executor, precisions, centers)
            ngeometries += length(uncached)
            nfailed += count(isnothing, precisions)
            for (index, center, precision) in zip(uncached, centers, precisions)
                if isnothing(precision)
                    component_index[index] = -1
                else
                    push!(components, _mw_gaussian(center, precision))
                    push!(multiplicities, 0)
                    push!(center_logp, logp[index])
                    component_index[index] = length(components)
                end
            end
            added = filter(>(0), component_index[indices])
            if isempty(added)
                adapting || break
                stop_reason, adapting = :geometry_failure, false
                continue
            end
            multiplicities[added] .+= 1
            ncomponent_proposals += length(added)
            q, center_logsum = _mw_center_mixture(components, center_logp, alg.executor,
                center_logsum, multiplicities, added)
            # Preserve the request for each selected occurrence before flooring.
            requested = floor.(Int, (probs(q)[added] ./ multiplicities[added]) .* npilot)
            drawn = 0
            for (index, count) in zip(added, requested)
                n = min(count, (fit_budget - nevals) ÷ draw_cost)
                n == 0 && continue
                batch = _mw_draw(components[index], logtarget, n, alg.executor, context)
                nevals += n
                fit_budget -= (draw_cost - 1) * n
                drawn += n
                append!(points, flatview(batch.v))
                append!(logp, batch.logp)
                append!(component_index, zeros(Int, n))
            end
            npilot = length(logp)
            push!(history, (; iteration, fresh = !adapting, npilot, drawn, ncomponents = length(q.components), ncomponent_proposals))
        end
        scores = logp .- first(_mw_pool_logpdf(q, components, points, score_cache, alg.executor))
        pilot_ess = isfinite(maximum(scores)) ? _mw_efficiency(scores).ess : zero(T)
    elseif alg.maxiter > 0
        stop_reason = :maxevals
    end
    q isa MixtureModel && !isnothing(alg.refit) && !isempty(fresh_logw) && (q = _mw_fit_mixture(q, fresh_points, fresh_logw, alg.refit, context))

    # Optional prior mixing does not affect the source discovery strategy.
    epsilon = T(alg.exploration_mass)
    if epsilon > 0
        ncomponent_proposals += Int(!has_prior)
        weights = (one(T) - epsilon) .* probs(q)
        q = if has_prior
            weights[1] += epsilon
            MixtureModel(q.components, weights)
        else
            MixtureModel(vcat([gprior], q.components), vcat(epsilon, weights))
        end
    end
    nproduction = something(alg.nsamples, max(alg.batchsize, npilot))
    pilot_efficiency = nothing
    if isfinite(alg.target_ess)
        pilot = _mw_draw(q, logtarget, alg.batchsize, alg.executor, context)
        nevals += alg.batchsize
        pilot_efficiency = _mw_efficiency(pilot.logp .- pilot.logr).efficiency
        limit = something(alg.nsamples, alg.maxevals - nevals)
        requested = alg.target_ess / pilot_efficiency
        nproduction = requested >= limit ? limit : ceil(Int, requested)
    end
    production = _mw_draw(q, logtarget, nproduction, alg.executor, context)
    nevals += nproduction
    logw = production.logp .- production.logr
    output = _mw_efficiency(alg.smooth_weights ? _mw_smooth(logw) : logw)
    diagnostic = isnothing(alg.weight_diagnostic) ? nothing : alg.weight_diagnostic(logw)
    smpls_z = DensitySampleVector(v = production.v, logd = production.logp, weight = output.weight)
    smpls = inverse(f).(smpls_z)
    dsm = DensitySampleMeasure(smpls, dof = dim, ess = output.ess)
    q_z = batmeasure(q)
    q_original = pushfwd(inverse(f), q_z)
    approx = alg.pretransform isa DoNotTransform ? BispacedMeasure(q_original) : BispacedMeasure(q_original, q_z, hash(f))
    result = (; stop_reason, niterations, nfresh, nevals, ngeometries, nfailed, nproduction,
        nseed_evals, nseed_exhausted, nhessians, npilot, pilot_ess, ncomponents = length(q.components), ncomponent_proposals,
        ess = output.ess, efficiency = output.efficiency, logweight_scale = output.logweight_scale,
        pareto_k = _mw_pareto_k(logw), max_weight = maximum(output.weight) / sum(output.weight),
        pilot_efficiency, diagnostic, history)
    return EvaluatedMeasure(em;
        transform_intent = alg.pretransform,
        f_transform = _viewrep_f(f, alg.pretransform),
        empirical = _viewrep_empirical(dsm, smpls_z, f, alg.pretransform, dim, output.ess),
        approx, samplegen = nothing, dof = dim,
        transformed = _viewrep_measure(transformed_m, alg.pretransform),
        evalinfo = MeasureEvalInfo(alg, result)
    )
end
