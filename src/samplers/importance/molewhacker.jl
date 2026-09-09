# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    MolewhackerSampling(; kwargs...)

Adaptive defensive Gaussian-mixture importance sampling. Fits local Fisher
Gaussians, validates proposal changes on fresh draws, then freezes the proposal
and draws fresh IID production samples. Requires a differentiable forward-model
likelihood and a standard-normal prior after `pretransform`.

Supports Normal, MvNormal, Poisson, Exponential, and product observation models.
Uses dense CPU geometry. Does not require MGVI. Optional `mode` accepts an
existing BAT optimization backend and maximizes the transformed target density.

Only production draws enter the returned empirical measure. Their weights are
`exp(logtarget - logproposal - logweight_scale)`, with the common scale retained
in `evalinfo.result`. The normalized proposal is stored in `approx`.
ESS measures weight concentration and does not establish mode coverage.

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct MolewhackerSampling{TR<:TransformIntent,IA<:InitvalAlgorithm,M,S,E<:BATExecutor,D} <: AbstractSamplingAlgorithm
    pretransform::TR = NormalBased()
    "Seed source in original coordinates. Used only when `nseeds > 0`."
    init::IA = InitFromTarget()
    "Number of initial seeds. Exact duplicate centers share a local Gaussian."
    nseeds::Int = 0
    "Optional density maximization backend for seeds and discovered centers."
    mode::M = nothing
    "Production count, or its cap when `target_ess` is finite."
    nsamples::Int = 10^4
    "Optional production ESS goal. A fresh pilot chooses the count before production."
    target_ess::Float64 = Inf
    "Draw count for each training, validation, and sizing batch."
    batchsize::Int = 1024
    maxiter::Int = 20
    "Maximum mixture size, including the prior."
    maxcomponents::Int = 32
    "Maximum target log-density calls, including mode searches and production. Geometry calls are separate."
    maxevals::Int = 10^5
    "Maximum separated candidate centers per training batch, independent of thread count."
    ncandidates::Int = 2
    "Minimum prior mixture mass, retained through all updates."
    exploration_mass::Float64 = 0.1
    "Variance multipliers assessed for each local Fisher Gaussian."
    covariance_scales::S = (1.0, 2.0, 4.0)
    "Minimum relative second-moment improvement on fresh validation draws."
    min_improvement::Float64 = 0.01
    "Stop after this many consecutive unaccepted training rounds."
    patience::Int = 3
    executor::E = default_executor()
    "Optional function of final log importance ratios, for example `MGVI.pareto_diagnostic`."
    weight_diagnostic::D = nothing
end
export MolewhackerSampling

function _mw_check(alg, context)
    @argcheck alg.nsamples > 0 && alg.batchsize > 0
    @argcheck alg.maxiter >= 0 && alg.nseeds >= 0
    @argcheck alg.maxcomponents > alg.nseeds
    @argcheck alg.ncandidates > 0 && alg.patience > 0
    @argcheck 0 < alg.exploration_mass < 1
    @argcheck 0 <= alg.min_improvement < 1
    @argcheck alg.target_ess > 0
    @argcheck !isempty(alg.covariance_scales) && all(s -> isfinite(s) && s > 0, alg.covariance_scales)
    pilot_count = isfinite(alg.target_ess) ? alg.batchsize : 0
    @argcheck alg.maxevals >= alg.nsamples && alg.maxevals - alg.nsamples >= pilot_count
    reserve = alg.nsamples + pilot_count
    @argcheck get_compute_unit(context) isa CPUnit
    T = get_precision(context)
    @argcheck 0 < T(alg.exploration_mass) < 1
    @argcheck all(s -> isfinite(T(s)) && T(s) > 0, alg.covariance_scales)
    return reserve
end

function _mw_scaled_gaussian(center::AbstractVector{T}, precision, scale) where T
    P = precision ./ T(scale)
    all(isfinite, P) || return nothing
    try
        return _mw_gaussian(center, P)
    catch err
        err isa Union{PosDefException,SingularException} || rethrow()
        return nothing
    end
end

function _mw_gaussian(center::AbstractVector{T}, precision) where T
    P = PDMat(Matrix{T}(precision))
    c = copy(center)
    return MvNormalCanon(c, P * c, P)
end

# All mixtures use the same concrete Gaussian type. Keep the prior first.
function _mw_mix(q, g, β, ε)
    T = eltype(first(q.components))
    β, ε = T(β), T(ε)
    w = (1 - β) .* probs(q)
    w[1] += β * ε
    push!(w, β * (1 - ε))
    components = vcat(q.components, [g])
    keep = findall(>(zero(T)), w)
    w, components = w[keep], components[keep]
    w[1] = max(w[1], ε)
    w[2:end] .*= (1 - w[1]) / sum(view(w, 2:length(w)))
    return MixtureModel(components, w)
end

function _mw_draw(q, logtarget, n, executor, context)
    v = VectorOfSimilarVectors(rand(get_rng(context), q, n))
    first_logp = logtarget(first(v))
    logp = Vector{typeof(float(first_logp))}(undef, n)
    logp[1] = first_logp
    exec_map!(logtarget, executor, view(logp, 2:n), view(v, 2:n))
    all(x -> isfinite(x) || x == -Inf, logp) || throw(ArgumentError("MolewhackerSampling encountered an invalid target log density."))
    logr = logpdf.(Ref(q), v)
    all(isfinite, logr) || throw(ArgumentError("MolewhackerSampling encountered a non-finite generating log density."))
    return (; v, logp, logr)
end

function _mw_loga(data)
    c = maximum(data.logp)
    isfinite(c) || return nothing
    return 2 .* (data.logp .- c) .- data.logr
end

function _mw_logobjective(loga, logq)
    return mapreduce(-, _logaddexp, loga, logq)
end

function _mw_fit_mass(loga, logq, logg)
    function objective(β)
        lq, lg = log1p(-β), log(β)
        return mapreduce(_logaddexp, eachindex(loga, logq, logg)) do i
            loga[i] - _logaddexp(lq + logq[i], lg + logg[i])
        end
    end
    # qβ = (q + g)/2 * (1 + (2β - 1)d), with d = (g - q)/(g + q).
    # Precompute scaled coefficients. The derivative then needs only arithmetic.
    T = promote_type(eltype(loga), eltype(logq), eltype(logg), Float64)
    a = map((v, q, g) -> T(v) - _logaddexp(T(q), T(g)), loga, logq, logg)
    a .= exp.(a .- maximum(a))
    d = map((q, g) -> tanh((T(g) - T(q)) / 2), logq, logg)
    lo, hi = 0.0, 1.0
    for _ in 1:32
        β = (lo + hi) / 2
        slope = sum(eachindex(a, d)) do i
            a[i] * d[i] / (1 + (2β - 1) * d[i])^2
        end
        if slope > 0
            lo = β
        else
            hi = β
        end
    end
    # d can round to ±1. Use the original log objective at endpoints.
    choices = (0.0, (lo + hi) / 2, 1.0)
    values = map(objective, choices)
    i = argmin(values)
    return choices[i], values[i]
end

struct MolewhackerBudgetReached <: Exception end

function _mw_mode(center, logtarget, mode, remaining, context)
    isnothing(mode) && return center, 0, false
    ncalls = Ref(0)
    f = x -> begin
        ncalls[] < remaining || throw(MolewhackerBudgetReached())
        ncalls[] += 1
        logtarget(x)
    end
    try
        r = maximize_density(f, center, mode, context)
        return collect(r.result), ncalls[], false
    catch err
        err isa MolewhackerBudgetReached || rethrow()
        return center, ncalls[], true
    end
end

function _mw_efficiency(logw)
    c = maximum(logw)
    isfinite(c) || throw(ArgumentError("MolewhackerSampling drew no finite positive target mass."))
    all(x -> isfinite(x) || x == -Inf, logw) || throw(ArgumentError("MolewhackerSampling encountered invalid importance ratios."))
    w = exp.(logw .- c)
    ess = sum(w)^2 / sum(abs2, w)
    return (; weight = w, ess, efficiency = ess / length(w), logweight_scale = c)
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

function evalmeasure_impl(em::EvaluatedMeasure, alg::MolewhackerSampling, context::BATContext)
    reserve = _mw_check(alg, context)
    transformed_m, f = transform_and_unshape(alg.pretransform, em, context)
    m = unevaluated(transformed_m)
    prior = getprior(m)
    is_std_mvnormal(prior) || throw(ArgumentError("MolewhackerSampling requires a standard-normal prior after pretransform."))
    model = ffcomp(_mw_model(getlikelihood(unevaluated(em))), inverse(f))
    logtarget = checked_logdensityof(m)
    T = get_precision(context)
    dim = some_dof(transformed_m)
    gprior = _mw_gaussian(zeros(T, dim), Matrix{T}(I, dim, dim))
    q = MixtureModel([gprior], [one(T)])
    ε = T(alg.exploration_mass)
    ad = alg.maxiter > 0 || alg.nseeds > 0 ? get_valid_adselector(context, alg) : get_adselector(context)
    fit_budget = alg.maxevals - reserve
    nevals, ngeometries, nfailed, niterations, stalled = 0, 0, 0, 0, 0
    stop_reason = :maxiter
    history = NamedTuple[]

    if alg.nseeds > 0
        initalg = apply_trafo_to_init(f, alg.init)
        seeds = bat_initval(transformed_m, alg.nseeds, initalg, context).result
        seed_components = typeof(gprior)[]
        seed_counts = Int[]
        failed_centers = Vector{Vector{T}}()
        for seed in seeds
            center, n, exhausted = _mw_mode(T.(collect(seed)), logtarget, alg.mode, fit_budget - nevals, context)
            nevals += n
            if exhausted
                stop_reason = :maxevals
                break
            end
            i = findfirst(g -> isequal(mean(g), center), seed_components)
            if !isnothing(i)
                seed_counts[i] += 1
                continue
            end
            any(c -> isequal(c, center), failed_centers) && continue
            P = _mw_local_precision(model, center, ad)
            ngeometries += 1
            if isnothing(P)
                nfailed += 1
                push!(failed_centers, copy(center))
                continue
            end
            push!(seed_components, _mw_gaussian(center, P))
            push!(seed_counts, 1)
        end
        if !isempty(seed_components)
            # Merge exact duplicate centers while preserving equal per-seed mass.
            weights = (one(T) - ε) .* T.(seed_counts) ./ T(sum(seed_counts))
            q = MixtureModel(vcat([gprior], seed_components), vcat(ε, weights))
        end
    end

    training = nothing
    for iteration in 1:alg.maxiter
        if fit_budget - nevals < alg.batchsize || fit_budget - nevals - alg.batchsize < alg.batchsize
            stop_reason = :maxevals
            break
        elseif length(q.components) >= alg.maxcomponents
            stop_reason = :maxcomponents
            break
        end
        niterations = iteration
        batch = _mw_draw(q, logtarget, alg.batchsize, alg.executor, context)
        nevals += alg.batchsize
        if isnothing(training)
            training = (v = Vector{Vector{T}}(batch.v), logp = copy(batch.logp), logr = copy(batch.logr))
        else
            append!(training.v, batch.v)
            append!(training.logp, batch.logp)
            append!(training.logr, batch.logr)
        end
        loga = _mw_loga(training)
        if isnothing(loga)
            push!(history, (; iteration, accepted = false, gain = zero(T), ncomponents = length(q.components)))
            stalled += 1
            if stalled >= alg.patience
                stop_reason = :no_improvement
                break
            end
            continue
        end
        logq = logpdf.(Ref(q), training.v)
        scores = training.logp .- logq
        order = sortperm(scores, rev = true)
        centers = Tuple{Vector{T},Matrix{T}}[]
        attempts = 0
        best, best_obj = q, _mw_logobjective(loga, logq)
        logprior = logpdf.(Ref(gprior), training.v)
        for idx in order
            attempts >= alg.ncandidates && break
            isfinite(scores[idx]) || continue
            center = training.v[idx]
            any(cp -> dot(center .- cp[1], cp[2] * (center .- cp[1])) < one(T), centers) && continue
            attempts += 1
            center, n, exhausted = _mw_mode(copy(center), logtarget, alg.mode, fit_budget - nevals - alg.batchsize, context)
            nevals += n
            if exhausted
                stop_reason = :maxevals
                break
            end
            any(cp -> dot(center .- cp[1], cp[2] * (center .- cp[1])) < one(T), centers) && continue
            P = _mw_local_precision(model, center, ad)
            ngeometries += 1
            if isnothing(P)
                nfailed += 1
            else
                # Exclude later centers inside this candidate's Fisher ellipsoid.
                push!(centers, (center, P))
                for scale in alg.covariance_scales
                    g = _mw_scaled_gaussian(center, P, scale)
                    if isnothing(g)
                        nfailed += 1
                        continue
                    end
                    logg = _logaddexp.(log(ε) .+ logprior, log1p(-ε) .+ logpdf.(Ref(g), training.v))
                    β, obj = _mw_fit_mass(loga, logq, logg)
                    if β > 0 && obj < best_obj
                        best, best_obj = _mw_mix(q, g, β, ε), obj
                    end
                end
            end
        end

        accepted, gain = false, zero(T)
        if best !== q
            # Candidates are now frozen. The comparison draw has its own denominator.
            r = MixtureModel(vcat(q.components, best.components), vcat(probs(q) ./ 2, probs(best) ./ 2))
            validation = _mw_draw(r, logtarget, alg.batchsize, alg.executor, context)
            nevals += alg.batchsize
            vala = _mw_loga(validation)
            if !isnothing(vala)
                old_obj = _mw_logobjective(vala, logpdf.(Ref(q), validation.v))
                new_obj = _mw_logobjective(vala, logpdf.(Ref(best), validation.v))
                gain = -expm1(new_obj - old_obj)
                accepted = gain > alg.min_improvement
                accepted && (q = best)
            end
        end
        push!(history, (; iteration, accepted, gain, ncomponents = length(q.components)))
        stop_reason == :maxevals && break
        stalled = accepted ? 0 : stalled + 1
        if stalled >= alg.patience
            stop_reason = :no_improvement
            break
        end
    end

    nproduction = alg.nsamples
    pilot_efficiency = nothing
    if isfinite(alg.target_ess)
        pilot = _mw_draw(q, logtarget, alg.batchsize, alg.executor, context)
        nevals += alg.batchsize
        pilot_efficiency = _mw_efficiency(pilot.logp .- pilot.logr).efficiency
        nproduction = ceil(Int, min(alg.nsamples, alg.target_ess / pilot_efficiency))
    end
    production = _mw_draw(q, logtarget, nproduction, alg.executor, context)
    nevals += nproduction
    logw = production.logp .- production.logr
    weights = _mw_efficiency(logw)
    diagnostic = isnothing(alg.weight_diagnostic) ? nothing : alg.weight_diagnostic(copy(logw))
    smpls_z = DensitySampleVector(v = production.v, logd = production.logp, weight = weights.weight)
    smpls = inverse(f).(smpls_z)
    dsm = DensitySampleMeasure(smpls, dof = dim, ess = weights.ess)
    q_z = batmeasure(q)
    q_original = pushfwd(inverse(f), q_z)
    approx = alg.pretransform isa DoNotTransform ? BispacedMeasure(q_original) : BispacedMeasure(q_original, q_z, hash(f))
    result = (; stop_reason, niterations, nevals, ngeometries, nfailed, nproduction,
        ncomponents = length(q.components), ess = weights.ess, efficiency = weights.efficiency,
        logweight_scale = weights.logweight_scale, pilot_efficiency, diagnostic, history)
    return EvaluatedMeasure(em;
        transform_intent = alg.pretransform,
        f_transform = _viewrep_f(f, alg.pretransform),
        empirical = _viewrep_empirical(dsm, smpls_z, f, alg.pretransform, dim, weights.ess),
        approx, samplegen = nothing, dof = dim,
        transformed = _viewrep_measure(transformed_m, alg.pretransform),
        evalinfo = MeasureEvalInfo(alg, result)
    )
end
