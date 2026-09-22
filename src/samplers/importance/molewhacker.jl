# This file is a part of BAT.jl, licensed under the MIT License (MIT).

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
@with_kw struct MolewhackerSampling{TR<:TransformIntent,IA,IM,E<:BATExecutor,D} <: AbstractSamplingAlgorithm
    pretransform::TR = NormalBased()
    "Seed source in original coordinates, or `nothing` for Sobol starts in normal coordinates."
    init::IA = nothing
    "Number of initial seeds. Zero skips mode initialization."
    nseeds::Int = 10
    "Seed optimizer. The default L-BFGS-B backend requires `import OptimizationLBFGSB`."
    init_mode::IM = nseeds == 0 ? nothing : OptimizationAlg(optalg = ext_default(pkgext(Val(:OptimizationLBFGSB)), Val(:LBFGSB_ALG)))
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
    "Number of candidate centers added per round."
    ncandidates::Int = Threads.nthreads()
    "Optional prior coefficient added after fitting. Zero leaves the fitted proposal unchanged."
    exploration_mass::Float64 = 0.0
    executor::E = default_executor()
    "Optional function of final log importance ratios."
    weight_diagnostic::D = nothing
end
export MolewhackerSampling

function _mw_check(alg, context)
    @argcheck (isnothing(alg.nsamples) || alg.nsamples > 0) && alg.batchsize > 0
    @argcheck alg.maxiter >= 0 && alg.nseeds >= 0 && alg.ncandidates > 0
    @argcheck alg.maxcomponents >= max(1, alg.nseeds + Int(alg.exploration_mass > 0))
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
    return reserve
end

function _mw_gaussian(center::AbstractVector{T}, precision) where T
    P = PDMat{T}(precision)
    c = copy(center)
    return MvNormalCanon(c, P * c, P)
end

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
        collect(maximize_density(counted, center, mode, context).result)
    end
    return exhausted ? center : result, ncalls, exhausted
end

function _mw_efficiency(logw)
    c = maximum(logw)
    isfinite(c) || throw(ArgumentError("MolewhackerSampling drew no finite positive target mass."))
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
    nevals, ngeometries, nfailed, nexhausted = 0, 0, 0, 0
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
            contexts = [set_rng(context, Philox4x((rand(get_rng(context), UInt64), UInt64(i)))) for i in 1:alg.nseeds]
            search = i -> _mw_mode(starts[i], logtarget, deepcopy(alg.init_mode), share + (i <= extra), contexts[i])
            exec_map!(search, alg.executor, results, collect(1:alg.nseeds))
        end
        nevals = sum(r -> r[2], results)
        nexhausted = count(r -> r[3], results)
        centers = [r[1] for r in results if !r[3]]
        precisions = Vector{Union{Nothing,typeof(gprior.J)}}(undef, length(centers))
        exec_map!(c -> _mw_local_precision(model, c, ad), alg.executor, precisions, centers)
        ngeometries = length(centers)
        keep = findall(!isnothing, precisions)
        nfailed = ngeometries - length(keep)
        components = typeof(gprior)[_mw_gaussian(centers[i], precisions[i]) for i in keep]
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
    return q, center_logp, center_logsum, nevals, ngeometries, nfailed, nexhausted
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
    q, center_logp, center_logsum, nevals, ngeometries, nfailed, nseed_exhausted =
        _mw_initial_proposal(transformed_m, f, model, logtarget, gprior, alg, fit_budget, ad, context)
    nseed_evals = nevals
    components = copy(q.components)
    multiplicities = ones(Int, length(components))
    ncomponent_proposals = length(components)
    has_prior = first(q.components) === gprior
    component_limit = alg.maxcomponents - Int(!has_prior && alg.exploration_mass > 0)
    # Automatic output reserves one fresh draw for each discovery point.
    draw_cost = isnothing(alg.nsamples) ? 2 : 1
    niterations, npilot = 0, 0
    stop_reason = nseed_exhausted > 0 && nseed_exhausted == alg.nseeds ? :maxevals : :maxiter
    history = NamedTuple[]
    pilot_ess = nothing

    if alg.maxiter > 0 && fit_budget - nevals >= alg.batchsize
        batch = _mw_draw(q, logtarget, alg.batchsize, alg.executor, context)
        nevals += alg.batchsize
        points, logp = Matrix{T}(flatview(batch.v)), batch.logp
        npilot = length(logp)
        # Zero marks unvisited points; -1 marks failed geometry. Positive entries
        # identify stored Gaussians, so reselection needs no equality search.
        component_index = zeros(Int, npilot)
        for iteration in 1:alg.maxiter
            scores = logp .- _mw_batched_logpdf(q, points, alg.executor)
            if !isfinite(maximum(scores))
                stop_reason = :no_finite_candidate
                break
            end
            # Recycled scores guide fitting only; they never become output weights.
            pilot_ess = _mw_efficiency(scores).ess
            if pilot_ess > alg.target_pool_ess
                stop_reason = :pilot_ess
                break
            elseif pilot_ess / npilot > alg.target_efficiency
                stop_reason = :pool_efficiency
                break
            elseif fit_budget - nevals < draw_cost
                stop_reason = :maxevals
                break
            elseif ncomponent_proposals >= component_limit
                stop_reason = :maxcomponents
                break
            end
            niterations = iteration
            nselected = min(alg.ncandidates, npilot, component_limit - ncomponent_proposals)
            indices = partialsortperm(scores, 1:nselected, rev = true)
            uncached = filter(i -> iszero(component_index[i]), indices)
            centers = collect.(eachcol(view(points, :, uncached)))
            precisions = Vector{Union{Nothing,typeof(gprior.J)}}(undef, length(uncached))
            exec_map!(c -> _mw_local_precision(model, c, ad), alg.executor, precisions, centers)
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
                stop_reason = :geometry_failure
                break
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
                points = hcat(points, Matrix{T}(flatview(batch.v)))
                append!(logp, batch.logp)
                append!(component_index, zeros(Int, n))
            end
            npilot = length(logp)
            push!(history, (; iteration, npilot, drawn, ncomponents = length(q.components), ncomponent_proposals))
        end
        scores = logp .- _mw_batched_logpdf(q, points, alg.executor)
        pilot_ess = isfinite(maximum(scores)) ? _mw_efficiency(scores).ess : zero(T)
    elseif alg.maxiter > 0
        stop_reason = :maxevals
    end

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
    output = _mw_efficiency(logw)
    diagnostic = isnothing(alg.weight_diagnostic) ? nothing : alg.weight_diagnostic(logw)
    smpls_z = DensitySampleVector(v = production.v, logd = production.logp, weight = output.weight)
    smpls = inverse(f).(smpls_z)
    dsm = DensitySampleMeasure(smpls, dof = dim, ess = output.ess)
    q_z = batmeasure(q)
    q_original = pushfwd(inverse(f), q_z)
    approx = alg.pretransform isa DoNotTransform ? BispacedMeasure(q_original) : BispacedMeasure(q_original, q_z, hash(f))
    result = (; stop_reason, niterations, nevals, ngeometries, nfailed, nproduction,
        nseed_evals, nseed_exhausted, npilot, pilot_ess, ncomponents = length(q.components), ncomponent_proposals,
        ess = output.ess, efficiency = output.efficiency, logweight_scale = output.logweight_scale,
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
