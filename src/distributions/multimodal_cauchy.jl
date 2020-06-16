struct MultimodalCauchy{M<:MixtureModel, R<:Real, N<:Int64, P<:Product}  <: ContinuousMultivariateDistribution
    bimodal::M
    σ::R
    n::N
    likelihood::P
end

"""
Multimodal Cauchy distribution as described in the AHMI paper.
Assumes a mixture of two modes.

BAT.MultimodalCauchy(μ=1.0,σ=0.2, n=4)

"""

function MultimodalCauchy(μ::Real, σ::Real, n::Int64)
    MM = MixtureModel([Cauchy(-μ, σ), Cauchy(μ, σ)])
    likelihood = _construct_likelihood(MM, σ, n)
    println(typeof(MM))
    println(typeof(σ))
    println(typeof(n))
    println(typeof(likelihood))
    MultimodalCauchy(MM, σ, n, likelihood)
end

Distributions.mean(d::MultimodalCauchy) = Distributions.mean(d.likelihood)
Distributions.var(d::MultimodalCauchy) = Distributions.var(d.likelihood)
Distributions.std(d::MultimodalCauchy) = Distributions.std(d.likelihood)

function Distributions.truncated(d::MultimodalCauchy, lb::Real, ub::Real)
    μ = d.bimodal.components[1].μ
    σ = d.bimodal.components[1].σ
    MM = MixtureModel([Distributions.truncated(Cauchy(-μ, σ), lb, ub), Distributions.truncated(Cauchy(μ, σ), lb, ub)])
    σ = d.σ
    n = d.n
    dists = vcat(MM, MM, [Distributions.truncated(Cauchy(0, σ), lb, ub) for i in 3:n])
    likelihood = product_distribution(dists)
    MultimodalCauchy(MM, σ, n, likelihood)
end

Base.size(d::MultimodalCauchy) = size(d.likelihood)
Base.length(d::MultimodalCauchy) = length(d.likelihood)
Base.eltype(d::MultimodalCauchy) = eltype(d.likelihood)

function Distributions._logpdf(d::MultimodalCauchy, x::AbstractArray)
    Distributions._logpdf(d.likelihood, x)
end

function Distributions._rand!(rng::AbstractRNG, d::MultimodalCauchy, x::AbstractVector)
    Distributions._rand!(rng, d.likelihood, x)
end

function _construct_likelihood(MM::MixtureModel, σ::Real, n::Int64)
    dists = vcat(MM, MM, [Cauchy(0, σ) for i in 3:n])
    likelihood = product_distribution(dists)
    return likelihood
end
