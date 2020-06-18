# This file is a part of BAT.jl, licensed under the MIT License (MIT).



struct FunnelDistribution <: ContinuousMultivariateDistribution
    a::Real
    b::Real
    n::Int
end

"""
    BAT.FunnelDistribution(a=1, b=0.5, n=3)

*BAT-internal, not part of stable public API.*

Funnel distribution (Caldwell et al.)[https://arxiv.org/abs/1808.08051].

# Arguments
- `a::Real`: Variance of the dominant normal distribution. Defaults to 1.
- `b::Real`: Variance of the supporting normal distributions. Defaults to 0.5.
- `n::Int`: Number of dimensions. Defaults to 3.
"""
function FunnelDistribution(;a::Real=1, b::Real=0.5, n::Int=3)
    a, b = promote(a, b)
    FunnelDistribution(a, b, n)
end

Base.length(d::FunnelDistribution) = d.n
Base.eltype(d::FunnelDistribution) = Base.eltype(d.a)

Distributions.mean(d::FunnelDistribution) = zeros(d.n)

function Distributions._logpdf(d::FunnelDistribution, x::AbstractArray)
    dist = _construct_dist(d.a, d.b, x)
    return Distributions._logpdf(dist, x)
end

function Distributions._rand!(rng::AbstractRNG, d::FunnelDistribution, x::AbstractVector)
    x[1] = rand(Normal(0, d.a^2))
    for i in 2:length(x)
        @inbounds x[i] = rand(Normal(0, exp(2*d.b*x[1])))
    end
    return x
end

_update_funnel(d::FunnelDistribution, λ::AbstractArray) = FunnelDistribution(d.a, d.b, λ)

function _construct_dist(a::Real, b::Real, λ::AbstractVector)
    n = length(λ)
    a = float(a)
    b = float(b)
    dist = Vector{Normal}(undef, n)
    dist[1] = Normal(0.0, a^2)
    for i in 2:n
        @inbounds dist[i] = Normal(0.0, exp(2*b*λ[1]))
    end
    return product_distribution(dist)
end
