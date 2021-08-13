# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    BAT.MultimodalStudentT <: Distribution{Multivariate,Continuous}

*Experimental feature, not part of stable public API.*

The Multimodal Student-t Distribution (It's defined similarly to multimodal Cauchy
defined in [Caldwell et al.](https://arxiv.org/abs/1808.08051)).

Assumes two bimodal peaks, each in its own dimension.

Constructors:

* ```BAT.MultimodalStudentT(; μ::Real=1, σ::Float64=0.2, ν::Integer=1, n::Integer=4)```

Arguments:

* `μ::Real`: The location parameter used for the two bimodal peaks.
* `σ::Float64`: The scale parameter shared among all components.
* `ν::Int`: The degrees of freedom.
* `n::Int`: The number of dimensions.

Fields:

$(TYPEDFIELDS)

!!! note

    All fields of `MultimodalStudentT` are considered internal and subject to
    change without deprecation.
"""
struct MultimodalStudentT{M<:MixtureModel, P<:Product} <: Distribution{Multivariate,Continuous}
    bimodals::M
    σ::Float64
    n::Int
    dist::P
end

function MultimodalStudentT(;μ::Real=1., σ::Float64=0.2, ν::Int=1, n::Integer=4)
    @argcheck n > 1 "Minimum number of dimensions for MultimodalCauchy is 2" 
    mixture_model = MixtureModel([LocationScale(-μ, σ, TDist(ν)), LocationScale(μ, σ, TDist(ν))])
    dist = _construct_dist(mixture_model, σ, ν, n)
    MultimodalStudentT(mixture_model, σ, n, dist)
end

Base.size(d::MultimodalStudentT) = size(d.dist)
Base.length(d::MultimodalStudentT) = length(d.dist)
Base.eltype(d::MultimodalStudentT) = eltype(d.dist)

Statistics.mean(d::MultimodalStudentT) = Distributions.mean(d.dist)
Statistics.var(d::MultimodalStudentT) = Distributions.var(d.dist)
Statistics.cov(d::MultimodalStudentT) = Distributions.cov(d.dist)

function StatsBase.params(d::MultimodalStudentT)#Check this
    (
        vcat(d.bimodals.components[1].μ, d.bimodals.components[2].μ, zeros(d.n-2)),
        [d.σ for i in 1:d.n],
        d.n
    )
end

function Distributions._logpdf(d::MultimodalStudentT, x::AbstractArray)
    convert(eltype(x), Distributions._logpdf(d.dist, x))::eltype(x)
end

function Distributions._rand!(rng::AbstractRNG, d::MultimodalStudentT, x::AbstractVector)
    Distributions._rand!(rng, d.dist, x)
end

function _construct_dist(mixture_model::MixtureModel, σ::Real, ν::Int, n::Integer)
    vector_of_dists = vcat(mixture_model, mixture_model, [LocationScale(0, σ, TDist(ν)) for i in 3:n])
    dist = product_distribution(vector_of_dists)
    return dist
end
