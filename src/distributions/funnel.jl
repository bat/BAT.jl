# This file is a part of BAT.jl, licensed under the MIT License (MIT).


@doc doc"""
    BAT.FunnelDistribution([a=1, b=0.5, n=3])

Funnel distribution (Caldwell et al.)[https://arxiv.org/abs/1808.08051].

# Arguments
- `a::Real`: Variance of the dominant normal distribution.
- `b::Real`: Variance of the supporting normal distributions.
- `n::Int`: Number of dimensions.

Constructors:

```julia
BAT.FunnelDistribution(a::Real, b::Real, n::Int)
```
"""
struct FunnelDistribution <: ContinuousMultivariateDistribution
    a::Real
    b::Real
    n::Int
end

function FunnelDistribution(;a::Real=1, b::Real=0.5, n::Integer=3)
    a, b = promote(a, b)
    FunnelDistribution(a, b, n)
end

Base.length(d::FunnelDistribution) = d.n
Base.eltype(d::FunnelDistribution) = Base.eltype(d.a)

Statistics.mean(d::FunnelDistribution) = zeros(d.n)

function Statistics.cov(dist::BAT.FunnelDistribution)
    cov(nestedview(rand(bat_determ_rng(), sampler(dist), 10^5)))
end

StatsBase.params(d::FunnelDistribution) = (d.a, d.b, d.n)


function Distributions._logpdf(d::FunnelDistribution, x::AbstractArray)
    idxs = eachindex(x)
    s = logpdf(Normal(0, d.a^2), x[idxs[1]])
    @inbounds for i in idxs[2:end]
        s += logpdf(Normal(0, exp(2 * d.b * x[1])), x[i])
    end
    s
end


function Distributions._rand!(rng::AbstractRNG, d::FunnelDistribution, x::AbstractVector)
    idxs = eachindex(x)
    x[idxs[1]] = rand(rng, Normal(0, d.a^2))
    @inbounds for i in idxs[2:end]
        x[i] = rand(rng, Normal(0, exp(2 * d.b * x[1])))
    end
    return x
end
