struct MultimodalCauchy{M<:MixtureModel, P<:Product}  <: ContinuousMultivariateDistribution
    bimodals::M
    σ::Float64
    n::Int
    dist::P
end

@doc doc"""
    BAT.MultimodalCauchy([μ=1, σ=0.5, n=3])

*BAT-internal, not part of stable public API.*

The Multimodal Cauchy Distribution (Caldwell et al.)[https://arxiv.org/abs/1808.08051].

Assumes two bimodal peaks, each in its own dimension.

# Arguments
- `μ::Real`: The location parameter used for the two bimodal peaks.
- `σ::Float64`: The scale parameter shared among all components.
- `n::Int`: The number of dimensions.
"""
function MultimodalCauchy end

function MultimodalCauchy(;μ::Real=1, σ::Float64=0.2, n::Int64=4)
    mixture_model = MixtureModel([Cauchy(-μ, σ), Cauchy(μ, σ)])
    dist = _construct_dist(mixture_model, σ, n)
    MultimodalCauchy(mixture_model, σ, n, dist)
end

Base.size(d::MultimodalCauchy) = size(d.dist)
Base.length(d::MultimodalCauchy) = length(d.dist)
Base.eltype(d::MultimodalCauchy) = eltype(d.dist)

Distributions.mean(d::MultimodalCauchy) = Distributions.mean(d.dist)
Distributions.var(d::MultimodalCauchy) = Distributions.var(d.dist)
Distributions.cov(d::MultimodalCauchy) = Distributions.cov(d.dist)

function Distributions._logpdf(d::MultimodalCauchy, x::AbstractArray)
    Distributions._logpdf(d.dist, x)
end

function Distributions._rand!(rng::AbstractRNG, d::MultimodalCauchy, x::AbstractVector)
    Distributions._rand!(rng, d.dist, x)
end

function _construct_dist(mixture_model::MixtureModel, σ::Real, n::Int64)
    vector_of_dists = vcat(mixture_model, mixture_model, [Cauchy(0, σ) for i in 3:n])
    dist = product_distribution(vector_of_dists)
    return dist
end
