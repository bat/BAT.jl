# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    BAT.MultimodalCauchy <: Distribution{Multivariate,Continuous}

*Experimental feature, not part of stable public API.*

The Multimodal Cauchy Distribution (see
[Caldwell et al.](https://arxiv.org/abs/1808.08051) for definition).

Assumes two bimodal peaks, each in its own dimension.

Constructors:

* ```BAT.MultimodalCauchy(; μ::Real=1, σ::Float64=0.2, n::Integer=4)```

Arguments:

* `μ::Real`: The location parameter used for the two bimodal peaks.
* `σ::Float64`: The scale parameter shared among all components.
* `n::Int`: The number of dimensions.

Fields:

$(TYPEDFIELDS)

!!! note

    All fields of `MultimodalCauchy` are considered internal and subject to
    change without deprecation.
"""
struct MultimodalCauchy{M<:MixtureModel, P<:Product} <: Distribution{Multivariate,Continuous}
    bimodals::M
    σ::Float64
    n::Int
    dist::P
end

function MultimodalCauchy(;μ::Real=1, σ::Float64=0.2, n::Integer=4)
    @argcheck n > 1 "Minimum number of dimensions for MultimodalCauchy is 2" 
    mixture_model = MixtureModel([Cauchy(-μ, σ), Cauchy(μ, σ)])
    dist = _construct_dist(mixture_model, σ, n)
    MultimodalCauchy(mixture_model, σ, n, dist)
end

Base.size(d::MultimodalCauchy) = size(d.dist)
Base.length(d::MultimodalCauchy) = length(d.dist)
Base.eltype(d::MultimodalCauchy) = eltype(d.dist)

Statistics.mean(d::MultimodalCauchy) = Distributions.mean(d.dist)
Statistics.var(d::MultimodalCauchy) = Distributions.var(d.dist)
Statistics.cov(d::MultimodalCauchy) = Distributions.cov(d.dist)

function StatsBase.params(d::MultimodalCauchy)
    (
        vcat(d.bimodals.components[1].μ, d.bimodals.components[2].μ, zeros(d.n-2)),
        [d.σ for i in 1:d.n],
        d.n
    )
end

function Distributions._logpdf(d::MultimodalCauchy, x::AbstractArray)
    convert(eltype(x), Distributions._logpdf(d.dist, x))::eltype(x)
end

function Distributions._rand!(rng::AbstractRNG, d::MultimodalCauchy, x::AbstractVector)
    Distributions._rand!(rng, d.dist, x)
end

function _construct_dist(mixture_model::MixtureModel, σ::Real, n::Integer)
    vector_of_dists = vcat(mixture_model, mixture_model, [Cauchy(0, σ) for i in 3:n])
    dist = product_distribution(vector_of_dists)
    return dist
end
