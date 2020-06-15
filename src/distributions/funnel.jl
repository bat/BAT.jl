struct Funnel{T<:Real, V<:AbstractVector, dimensions<:Int64, L<:ContinuousMultivariateDistribution} <: ContinuousMultivariateDistribution
    a::T
    b::T
    λ::V
    n::dimensions
    dists
    likelihood::L
end

function Funnel(a::Real, b::Real, λ::AbstractVector)
    n = length(λ)
    λ = float(λ)
    dists = _construct_dists(a, b, λ)
    pd = product_distribution(dists)
    Funnel(a, b, λ, n, dists, pd)
end

Base.length(d::Funnel) = d.n
Base.eltype(d::Funnel) = eltype(d.λ)

Distributions.mean(d::Funnel) = Distributions.mean(d.likelihood)
Distributions.var(d::Funnel) = Distributions.var(d.likelihood)
Distributions.std(d::Funnel) = Distributions.std(d.likelihood)

## Maybe this is incorrect, because x should be a variable?
##Distributions._logpdf(d::Funnel, x::AbstractArray) = Distributions._logpdf(d.likelihood, x)
# This has to be updated for bat_sample to work correctly, but then it messes with the integration
function Distributions._logpdf(d::Funnel, x::AbstractArray)
    likelihood = product_distribution(_construct_dists(d.a, d.b, x))
    return Distributions._logpdf(likelihood, x)
end
#function Distributions._logpdf(d::Funnel, x::AbstractArray)
#    d_updated = _update_funnel(d, x)
#    return Distributions._logpdf(d_updated.likelihood, x)
#end

## Do the same method as logpdf, but now we x isn't directly given apriori
##Distributions._rand!(rng::AbstractRNG, d::Funnel, x::AbstractVector) = Distributions._rand!(rng, d.likelihood, x)
#function Distributions._rand!(rng::AbstractRNG, d::Funnel, x::AbstractVector)
#    rsample = Distributions._rand!(rng, d.likelihood, x)
#    likelihood = product_distribution(_construct_dists(d.a, d.b, rsample))
#    rsample = Distributions._rand!(rng, likelihood, rsample)
#    for i in eachindex(x)
#        x[i] = rsample[i]
#    end
#    return x
#end
function Distributions._rand!(rng::AbstractRNG, d::Funnel, x::AbstractVector)
    x[1] = rand(Normal(0, d.a^2))
    for i in 2:length(x)
        @inbounds x[i] = rand(Normal(0, exp(2*d.b*x[1])))
    end
    return x
end

_update_funnel(d::Funnel, λ::AbstractArray) = Funnel(d.a, d.b, λ)

function Distributions.truncated(d::Funnel, l::Real, u::Real)
    truncated_dists = truncated.(d.dists, float(l), float(u))
    pd = product_distribution(truncated_dists)
    return Funnel(d.a, d.b, d.λ, d.n, truncated_dists, pd)
end

function _construct_dists(a::Real, b::Real, λ::AbstractVector)
    n = length(λ)
    a = float(a)
    b = float(b)
    dists = Vector{Normal}(undef, n)
    dists[1] = Normal(0.0, a^2)
    let b=b, dists=dists, λ=λ, n=n
        for i in 2:n
            dists[i] = Normal(0.0, exp(2*b*λ[1]))
        end
    end
    return dists
end

function _likelihood(a::Real, b::Real, λ::AbstractArray)
    # Normal{Real} doesn't work here
    a = float(a)
    b = float(b)
    λ = float(λ)
    dists = _construct_dists(a, b, λ)
    likelihood = product_distribution(dists)
    return likelihood
end
