# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    StandardUvNormal{T<:Real} <: Distributions.Distribution{Univariate,Continuous}

A standard normal distribution with a mean of zero and a variance of one.

Constructor:
```
    StandardUvNormal()
    StandardUvNormal{T<:Real}()
```
"""
struct StandardUvNormal{T<:Real} <: Distributions.Distribution{Univariate,Continuous}
end

StandardUvNormal() = StandardUvNormal{Float64}()

Distributions.Normal(d::StandardUvNormal{T}) where T = Normal{T}(zero(T), one(T))
Base.convert(::Type{Distributions.Normal}, d::StandardUvNormal) = Normal(d)


Base.minimum(d::StandardUvNormal{T}) where T = convert(float(T), -Inf)
Base.maximum(d::StandardUvNormal{T}) where T = convert(float(T), +Inf)

StatsBase.params(d::StandardUvNormal) = (mean(d), var(d))
@inline Distributions.partype(d::StandardUvNormal{T}) where {T<:Real} = T

Distributions.location(d::StandardUvNormal) = mean(d)
Distributions.scale(d::StandardUvNormal) = var(d)

Base.eltype(::Type{StandardUvNormal{T}}) where T = T

Statistics.mean(d::StandardUvNormal{T}) where T = zero(T)
StatsBase.median(d::StandardUvNormal{T}) where T = zero(T)
StatsBase.mode(d::StandardUvNormal{T}) where T = zero(T)
StatsBase.modes(d::StandardUvNormal{T}) where T = FillArrays.Zeros{T}(1)

Statistics.var(d::StandardUvNormal{T}) where T = one(T)
StatsBase.std(d::StandardUvNormal{T}) where T = one(T)
StatsBase.skewness(d::StandardUvNormal{T}) where {T<:Real} = zero(T)
StatsBase.kurtosis(d::StandardUvNormal{T}) where {T<:Real} = zero(T)

StatsBase.entropy(d::StandardUvNormal) = entropy(Normal(d))

Distributions.logpdf(d::StandardUvNormal, x::Real) = logpdf(Normal(d), x)
Distributions.pdf(d::StandardUvNormal, x::Real) = pdf(Normal(d), x)
Distributions.logcdf(d::StandardUvNormal, x::Real) = logcdf(Normal(d), x)
Distributions.cdf(d::StandardUvNormal, x::Real) = cdf(Normal(d), x)
Distributions.logccdf(d::StandardUvNormal, x::Real) = logccdf(Normal(d), x)
Distributions.ccdf(d::StandardUvNormal, x::Real) = ccdf(Normal(d), x)
Distributions.quantile(d::StandardUvNormal, p::Real) = quantile(Normal(d), p)
Distributions.cquantile(d::StandardUvNormal, p::Real) = cquantile(Normal(d), p)
Distributions.mgf(d::StandardUvNormal, t::Real) = mgf(Normal(d), t)
Distributions.cf(d::StandardUvNormal, t::Real) = cf(Normal(d), t)

Distributions.gradlogpdf(d::StandardUvNormal, x::Real) = gradlogpdf(Normal(d), x)

Base.rand(rng::AbstractRNG, d::StandardUvNormal{T}) where T = randn(rng, float(T))

Distributions.truncated(d::StandardUvNormal, l::Real, u::Real) = truncated(Normal(d), l, u)

function Distributions.product_distribution(dists::AbstractVector{StandardUvNormal{T}}) where T
    StandardMvNormal{T}(length(eachindex(dists)))
end


"""
    StandardMvNormal{T<:Real} <: Distributions.AbstractMvNormal

A standard `n`-dimensional multivariate normal distribution with it's mean at
the origin and an identity covariance matrix.

Constructor:
```
    StandardMvNormal(n::Integer)
    StandardMvNormal{T<:Real}(n::Integer)
```
"""
struct StandardMvNormal{T<:Real} <: Distributions.AbstractMvNormal
    _dim::Int
end

StandardMvNormal(dim::Integer) = StandardMvNormal{Float64}(dim)

Distributions.Product(d::StandardMvNormal{T}) where T = Distributions.Product(Fill(StandardUvNormal{T}(), length(d)))
Base.convert(::Type{Distributions.Product}, d::StandardMvNormal) = Distributions.Product(d)

Distributions.MvNormal(d::StandardMvNormal{T}) where T = MvNormal(ScalMat(length(d), one(T)))
Base.convert(::Type{Distributions.MvNormal}, d::StandardMvNormal) = MvNormal(d)


function Distributions.insupport(d::StandardMvNormal, x::AbstractVector)
    length(d) == length(eachindex(x)) && all(isfinite, x)
end

Base.eltype(::Type{StandardMvNormal{T}}) where T = T

Base.length(d::StandardMvNormal{T}) where T = d._dim

@inline function Base.view(d::StandardMvNormal{T}, i::Integer) where T
    Base.@boundscheck Base.checkindex(Bool, Base.OneTo(length(d)), i) || Base.throw_boundserror(d, i)
    StandardUvNormal{T}()
end

function Base.view(d::StandardMvNormal{T}, I::AbstractArray{<:Integer}) where T
    Base.@boundscheck Base.checkindex(Bool, Base.OneTo(length(d)), I) || Base.throw_boundserror(d, I)
    StandardMvNormal{T}(length(eachindex(I)))
end


StatsBase.params(d::StandardMvNormal) = (mean(d), cov(d))
@inline Distributions.partype(d::StandardMvNormal{T}) where {T<:Real} = T

Statistics.mean(d::StandardMvNormal{T}) where T = FillArrays.Zeros{T}(length(d))
Statistics.var(d::StandardMvNormal{T}) where T = FillArrays.Ones{T}(length(d))
Statistics.cov(d::StandardMvNormal{T}) where T = Diagonal(var(d))

StatsBase.mode(d::StandardMvNormal) = mean(d)
StatsBase.modes(d::StandardMvNormal) = Fill(mean(d), 1)

Distributions.invcov(d::StandardMvNormal{T}) where T = cov(d)
Distributions.logdetcov(d::StandardMvNormal{T}) where T = zero(T)

StatsBase.entropy(d::StandardMvNormal) = entropy(MvNormal(d))

Distributions.logpdf(d::StandardMvNormal, x::AbstractVector{<:Real}) = logpdf(MvNormal(d), x)
Distributions.pdf(d::StandardMvNormal, x::AbstractVector{<:Real}) = pdf(MvNormal(d), x)
Distributions.sqmahal(d::StandardMvNormal, x::AbstractVector{<:Real}) = sqmahal(MvNormal(d), x)
# Distributions.sqmahal!(r::AbstractVector{<:Real}, d::StandardMvNormal, x::AbstractMatrix{<:Real}) = sqmahal!(r, MvNormal(d), x)

Distributions.gradlogpdf(d::StandardMvNormal, x::AbstractVector{<:Real}) = Distributions.gradlogpdf(MvNormal(d), x)

function Distributions._rand!(rng::AbstractRNG, d::StandardMvNormal, A::AbstractVector{T}) where {T<:Real}
    broadcast!(x -> randn(rng, T), A, A)
end

function Distributions._rand!(rng::AbstractRNG, d::StandardMvNormal, A::AbstractMatrix{T}) where {T<:Real}
    broadcast!(x -> randn(rng, T), A, A)
end
