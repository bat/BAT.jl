# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type MultiChainConvergenceTest <: ConvergenceTest end


function check_convergence!(
    chains::AbstractVector{<:MCMCIterator},
    samples::AbstractVector{<:DensitySampleVector},
    algorithm::ConvergenceTest,
    context::BATContext
)
    result = convert(Bool, bat_convergence(samples, algorithm, context).result)
    for chain in chains
        chain.info = MCMCChainStateInfo(chain.info, converged = result)
    end
    result
end

function check_convergence!(
    mcmc_states::AbstractVector{<:MCMCState}, 
    samples::AbstractVector{<:DensitySampleVector}, 
    algorithm::ConvergenceTest, 
    context::BATContext
)
    chain_states = getfield.(mcmc_states, :chain_state)
    check_convergence!(chain_states, samples, algorithm, context)
end


"""
    gr_Rsqr(stats::AbstractVector{<:MCMCBasicStats})
    gr_Rsqr(samples::AbstractVector{<:DensitySampleVector})

*BAT-internal, not part of stable public API.*

Gelman-Rubin ``\$R^2\$`` for all DOF.

See [A. Gelman and D. B. Rubin, "Inference from Iterative Simulation
Using Multiple Sequences"
(1992)](https://doi.org/10.1214/ss/1177011136).
"""
function gr_Rsqr end

function gr_Rsqr(stats::AbstractVector{<:MCMCBasicStats})
    m = _stats_dof(first(stats))
    W = mean([cs.param_stats.cov[i,i] for cs in stats, i in 1:m], dims=1)[:]
    B = var([cs.param_stats.mean[i] for cs in stats, i in 1:m], dims=1)[:]
    (W .+ B) ./ W
end

function gr_Rsqr(samples::AbstractVector{<:DensitySampleVector})
    gr_Rsqr(MCMCBasicStats.(samples))
end



"""
    struct GelmanRubinConvergence <: ConvergenceTest

Gelman-Rubin maximum R^2 convergence test.

See [A. Gelman and D. B. Rubin, "Inference from Iterative Simulation
Using Multiple Sequences"
(1992)](https://doi.org/10.1214/ss/1177011136).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct GelmanRubinConvergence <: MultiChainConvergenceTest
    threshold::Float64 = 1.1
end

export GelmanRubinConvergence

function bat_convergence_impl(samples::AbstractVector{<:DensitySampleVector}, algorithm::GelmanRubinConvergence, ::BATContext)
    max_Rsqr = maximum(gr_Rsqr(samples))
    vt = ValueAndThreshold{max_Rsqr}(max_Rsqr, <=, algorithm.threshold)
    converged = convert(Bool, vt)
    @debug begin
        success_str = converged ? "have" : "have *not*"
        "Chains $success_str converged, max(R^2) = $(vt.value), threshold = $(vt.threshold)"
    end
    (result = vt,)
end



@doc doc"""
    bg_R_2sqr(stats::AbstractVector{<:MCMCBasicStats}; corrected::Bool = false)
    bg_R_2sqr(samples::AbstractVector{<:DensitySampleVector}; corrected::Bool = false)

*BAT-internal, not part of stable public API.*

Brooks-Gelman R_2^2 for all DOF.
If normality is assumed, 'corrected' should be set to true to account for the sampling variability.

See [S. P. Brooks and A. Gelman, "General Methods for Monitoring
Convergence of Iterative Simulations"
(1998)](https://doi.org/10.1080/10618600.1998.10474787); the
`corrected` option implements the paper's df-corrected variant.
"""
function bg_R_2sqr(stats::AbstractVector{<:MCMCBasicStats}; corrected::Bool = false)
    p = _stats_dof(first(stats))
    m = length(stats)
    n = mean(Float64.(nsamples.(stats)))

    σ_W = var([cs.param_stats.cov[i,i] for cs in stats, i in 1:p], dims = 1)[:]
    B  = var([cs.param_stats.mean[i] for cs in stats, i in 1:p], dims = 1)[:]
    W = mean([cs.param_stats.cov[i,i] for cs in stats, i in 1:p], dims = 1)[:]

    σ_sq = m * (n - 1) / (m*n - 1) * W + n * (m - 1) / (m*n - 1) * B

    R_unc = σ_sq ./ W

    if corrected == false
        return R_unc
    end

    σ_ij = [cs.param_stats.cov[i,i] for cs in stats, i in 1:p]
    x_ij = [cs.param_stats.mean[i] for cs in stats, i in 1:p]

    cov_σx = [cov(σ_ij[:,j], x_ij[:,j]) for j in 1:p]
    cov_σx_sq = [cov(σ_ij[:,j], x_ij[:,j].^2) for j in 1:p]

    N = (n-1)/n
    M = (m-1)/m
    V = N*σ_sq + M*B

    σ_V = N^2/m*σ_W + 2*M/(m-1)*B.^2 + 2*M*N/m*(cov_σx_sq - 2*B.*cov_σx)
    d = 2 * V.^2 ./ σ_V

    R_unc.*(d.+3)./(d.+1)
end

function bg_R_2sqr(samples::AbstractVector{<:DensitySampleVector}; corrected::Bool = false)
    bg_R_2sqr(MCMCBasicStats.(samples), corrected = corrected)
end



"""
    struct BrooksGelmanConvergence <: ConvergenceTest

Brooks-Gelman maximum R^2 convergence test.

See [S. P. Brooks and A. Gelman, "General Methods for Monitoring
Convergence of Iterative Simulations"
(1998)](https://doi.org/10.1080/10618600.1998.10474787).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct BrooksGelmanConvergence <: MultiChainConvergenceTest
    threshold::Float64 = 1.1
    corrected::Bool = false
end

export BrooksGelmanConvergence

function bat_convergence_impl(samples::AbstractVector{<:DensitySampleVector}, algorithm::BrooksGelmanConvergence, ::BATContext)
    max_Rsqr = maximum(bg_R_2sqr(samples, corrected = algorithm.corrected))
    vt = ValueAndThreshold{max_Rsqr}(max_Rsqr, <=, algorithm.threshold)
    converged = convert(Bool, vt)
    @debug begin
        success_str = converged ? "have" : "have *not*"
        "Chains $success_str converged, max(R^2) = $(vt.value), threshold = $(vt.threshold)"
    end
    (result = vt,)
end



function _rhat_walker_indices(chain::DensitySampleVector)
    idxs = findall(>(0), chain.weight)
    eltype(chain.info) <: MCMCSampleID || return [1 => idxs]
    ids = sort!(unique(id.walkerid for id in view(chain.info, idxs)))
    [id => sort!([i for i in idxs if chain.info[i].walkerid == id];
        by = i -> (chain.info[i].chaincycle, chain.info[i].stepno)) for id in ids]
end

function _rhat_split_runs(chains)
    counts = (chain -> accumulate(Base.Checked.checked_add, Int[0; chain.weight])).(chains)
    n = last(first(counts))
    all(c -> last(c) == n, counts) ||
        throw(ArgumentError("Rank-normalized R-hat requires equal draw counts per chain."))
    n >= 4 || return nothing
    half = n ÷ 2
    # Clip repetition runs at the split boundaries, omitting an odd middle draw.
    map([(i, offset) for i in eachindex(chains) for offset in (0, n - half)]) do (i, offset)
        weight = diff(clamp.(counts[i] .- offset, 0, half))
        idxs = findall(>(0), weight)
        (; v = view(chains[i].v, idxs), weight = weight[idxs])
    end
end

function _rank_normalize(values, weights, order = sortperm(values))
    total = sum(weights)
    scores = Vector{Float64}(undef, length(values))
    sorted = view(values, order)
    before = 0
    i = 1
    while i <= length(order)
        j = something(findnext(!=(sorted[i]), sorted, i + 1), length(order) + 1)
        mass = sum(view(weights, view(order, i:j-1)))
        after = total - before - mass
        # Use the nearer tail so large repetition counts cannot round a quantile to one.
        p = (min(before, after) + mass / 2 + 1 / 8) / (total + 1 / 4)
        score = quantile(Normal(), p) * (before <= after ? 1 : -1)
        scores[view(order, i:j-1)] .= score
        before += mass
        i = j
    end
    scores
end

function _split_rhat(values, weights, ranges, n)
    stats = mean_and_var.(view.(Ref(values), ranges), FrequencyWeights.(view.(Ref(weights), ranges)); corrected = true)
    sqrt((n - 1) / n + var(first.(stats)) / mean(last.(stats)))
end

function _rank_normalized_rhat(chains::AbstractVector{<:DensitySampleVector})
    all(chain -> all(v -> all(isfinite, v), chain.v), chains) || return NaN
    splits = _rhat_split_runs(chains)
    isnothing(splits) && return NaN
    weights = reduce(vcat, getproperty.(splits, :weight))
    foldl(Base.Checked.checked_add, weights; init = 0)
    ends = cumsum(length.(getproperty.(splits, :weight)))
    ranges = UnitRange.([1; ends[1:end-1] .+ 1], ends)
    n = sum(first(splits).weight)
    maximum(1:totalndof(varshape(first(chains)))) do parameter
        values = reduce(vcat, (part -> getindex.(part.v, parameter)).(splits))
        order = sortperm(values)
        counts = cumsum(view(weights, order))
        half = last(counts) ÷ 2
        i = searchsortedfirst(counts, half)
        midpoint = float(values[order[i]])
        counts[i] == half && (midpoint = midpoint / 2 + values[order[i + 1]] / 2)
        folded = abs.(values .- midpoint)
        all(isfinite, folded) || (folded = abs.(values ./ 2 .- midpoint / 2))
        max(
            _split_rhat(_rank_normalize(values, weights, order), weights, ranges, n),
            _split_rhat(_rank_normalize(folded, weights), weights, ranges, n),
        )
    end
end


"""
    RankNormalizedRhatConvergence(; threshold = 1.01)

Rank-normalized and folded split R-hat convergence test from
[Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221).

Compare independent chains, treating integer weights as repetition counts.
For multiple walkers, compare matching walker IDs across chains. Return the
largest R-hat across parameters and walkers.

Chains must have matching parameters and equal draw counts. MCMC sample
IDs define draw order. Fewer than four draws or nonfinite draws return `NaN`,
which does not establish convergence.
"""
@with_kw struct RankNormalizedRhatConvergence <: MultiChainConvergenceTest
    threshold::Float64 = 1.01
end

export RankNormalizedRhatConvergence

function bat_convergence_impl(samples::AbstractVector{<:DensitySampleVector}, algorithm::RankNormalizedRhatConvergence, ::BATContext)
    length(samples) >= 2 || throw(ArgumentError("Rank-normalized R-hat requires at least two chains."))
    parameters = all_active_names(varshape(first(samples)))
    all(chain -> all_active_names(varshape(chain)) == parameters, samples) ||
        throw(ArgumentError("Rank-normalized R-hat requires matching parameters."))
    all(chain -> all(w -> w >= 0 && isinteger(w), chain.weight), samples) ||
        throw(ArgumentError("Rank-normalized R-hat requires nonnegative integer-valued weights."))
    walkers = _rhat_walker_indices.(samples)
    ids = first.(first(walkers))
    all(w -> first.(w) == ids, walkers) ||
        throw(ArgumentError("Rank-normalized R-hat requires matching walker IDs."))
    # Each comparison uses one walker from each independent sampler chain.
    max_rhat = isempty(ids) ? NaN : maximum(eachindex(ids)) do i
        chains = ((chain, w) -> unshaped.(chain[last(w[i])])).(samples, walkers)
        _rank_normalized_rhat(chains)
    end
    vt = ValueAndThreshold{:max_rhat}(max_rhat, <=, algorithm.threshold)
    @debug "Rank-normalized R-hat" converged=Bool(vt) max_rhat threshold=vt.threshold
    (result = vt,)
end



function bat_convergence_impl(samples::DensitySampleVector, algorithm::MultiChainConvergenceTest, context::BATContext)
    # create a vector of chains
    chains_ind = unique([i.chainid for i in samples.info])
    vector_chains = DensitySampleVector[]
    # ToDo: Improve implementation
    for i in chains_ind
        mask_chain = [j.chainid == i for j in samples.info]
        push!(vector_chains, samples[mask_chain])
    end

    bat_convergence_impl(vector_chains, algorithm, context)
end
