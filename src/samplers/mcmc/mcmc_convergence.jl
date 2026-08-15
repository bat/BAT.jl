# This file is a part of BAT.jl, licensed under the MIT License (MIT).


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

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct GelmanRubinConvergence <: ConvergenceTest
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

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct BrooksGelmanConvergence <: ConvergenceTest
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



function _rank_normalized_draws(samples::AbstractVector{<:DensitySampleVector})
    length(samples) >= 2 || throw(ArgumentError("Rank-normalized R-hat requires at least two chains."))
    samples = mapreduce(vcat, samples) do chain
        if !isempty(chain) && eltype(chain.info) <: MCMCSampleID
            return [
                unshaped.(chain[[info.walkerid == walkerid for info in chain.info]])
                for walkerid in unique(getfield.(chain.info, :walkerid))
            ]
        end
        [unshaped.(chain)]
    end

    draws = map(samples) do chain
        all(w -> w >= 0 && isinteger(w), chain.weight) ||
            throw(ArgumentError("Rank-normalized R-hat requires nonnegative integer-valued weights."))
        [sample.v for sample in chain for _ in 1:Int(sample.weight)]
    end

    draw_count = length(first(draws))
    all(length(chain) == draw_count for chain in draws) ||
        throw(ArgumentError("Rank-normalized R-hat requires equal draw counts per chain."))
    draw_count >= 4 ||
        throw(ArgumentError("Rank-normalized R-hat requires at least four draws per chain."))

    draws
end

function _rank_normalize(values::AbstractVector{<:Real})
    ranks = tiedrank(values)
    quantiles = (ranks .- 3 / 8) ./ (length(ranks) + 1 / 4)
    quantile.(Normal(), quantiles)
end

function _split_rhat(values::AbstractVector{<:AbstractVector{<:Real}})
    draw_count = length(first(values))
    within = mean(var.(values))
    between = draw_count * var(mean.(values))
    sqrt(((draw_count - 1) / draw_count * within + between / draw_count) / within)
end

function _split_chains(values::AbstractVector{<:Real}, draw_count::Integer, nchains::Integer)
    half_count = draw_count ÷ 2
    chains = reshape(values, draw_count, nchains)
    vcat(
        [view(chains, 1:half_count, i) for i in 1:nchains],
        [view(chains, draw_count - half_count + 1:draw_count, i) for i in 1:nchains],
    )
end

function _rank_normalized_rhat(draws::AbstractVector{<:AbstractVector})
    draw_count = length(first(draws))
    nchains = length(draws)
    nparams = length(first(first(draws)))

    maximum(1:nparams) do parameter
        values = reduce(vcat, (getindex.(chain, parameter) for chain in draws))
        normalized = _rank_normalize(values)
        folded = _rank_normalize(abs.(values .- median(values)))
        max(
            _split_rhat(_split_chains(normalized, draw_count, nchains)),
            _split_rhat(_split_chains(folded, draw_count, nchains)),
        )
    end
end


"""
    struct RankNormalizedRhatConvergence <: ConvergenceTest

Rank-normalized split R-hat convergence test.

For each parameter, splits each MCMC trajectory into two halves and computes

```math
\\widehat{R} = \\max\\left(
    \\widehat{R}_{\\mathrm{split}}(\\operatorname{ranknorm}(x)),
    \\widehat{R}_{\\mathrm{split}}(\\operatorname{ranknorm}(|x - \\operatorname{median}(x)|))
\\right).
```

The default convergence threshold is `1.01`. Sample weights must be
nonnegative integers; a sample with weight `w` is treated as `w` repeated
draws.

This is the rank-normalized and folded split R-hat diagnostic of
[Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct RankNormalizedRhatConvergence <: ConvergenceTest
    threshold::Float64 = 1.01
end

export RankNormalizedRhatConvergence

function bat_convergence_impl(samples::AbstractVector{<:DensitySampleVector}, algorithm::RankNormalizedRhatConvergence, ::BATContext)
    max_rhat = _rank_normalized_rhat(_rank_normalized_draws(samples))
    vt = ValueAndThreshold{max_rhat}(max_rhat, <=, algorithm.threshold)
    converged = convert(Bool, vt)
    @debug begin
        success_str = converged ? "have" : "have *not*"
        "Chains $success_str converged, max(rank-normalized R-hat) = $(vt.value), threshold = $(vt.threshold)"
    end
    (result = vt,)
end



function bat_convergence_impl(samples::DensitySampleVector, algorithm::Union{GelmanRubinConvergence, BrooksGelmanConvergence, RankNormalizedRhatConvergence}, context::BATContext)
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
