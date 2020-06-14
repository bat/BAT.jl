struct MultimodalCauchy{T<:AbstractVector, MM<:MixtureModel, P<:Product}  <: ContinuousMultivariateDistribution
    dists::T
    MM1::MM
    MM2::MM
    n::Int64
    likelihood::P
end

"""
Multimodal Cauchy distribution as described in the AHMI paper.
Assumes a mixture of two modes.

mixture = [Cauchy(-1., 0.2), Cauchy(1., 0.2)]
BAT.MultimodalCauchy(mixture, n=4)

"""

MultimodalCauchy(dists::AbstractVector) = MultimodalCauchy(dists, 4)

function MultimodalCauchy(dists::AbstractVector, n::Int64)
    MM1 = MixtureModel([dists[1], dists[2]])
    MM2 = MixtureModel([dists[1], dists[2]])
    likelihood = _construct_likelihood(MM1, MM2, n)
    MultimodalCauchy(dists, MM1, MM2, n, likelihood)
end

Distributions.mean(d::MultimodalCauchy) = Distributions.mean(d.likelihood)
Distributions.var(d::MultimodalCauchy) = Distributions.var(d.likelihood)
Distributions.std(d::MultimodalCauchy) = Distributions.std(d.likelihood)

Base.size(d::MultimodalCauchy) = size(d.likelihood)
Base.length(d::MultimodalCauchy) = length(d.likelihood)
Base.eltype(d::MultimodalCauchy) = eltype(d.likelihood)

function Distributions._logpdf(d::MultimodalCauchy, x::AbstractArray)
    Distributions._logpdf(d.likelihood, x)
end

function Distributions.truncated(d::MultimodalCauchy, l::Real, u::Real)
    trunc_dists = Distributions.truncated.(d.dists, l, u)
    MultimodalCauchy(trunc_dists, d.MM1, d.MM2, d.n, _construct_likelihood(d.MM1, d.MM2, d.n))
end

function Distributions._rand!(rng::AbstractRNG, d::MultimodalCauchy, x::AbstractVector)
    Distributions._rand!(rng, d.likelihood, x)
end

function _construct_likelihood(MM1::MixtureModel, MM2::MixtureModel, n::Int64)
    dists = vcat(MM1, MM2, [Cauchy(0, 0.2) for i in 3:n])
    likelihood = product_distribution(dists)
    return likelihood
end

