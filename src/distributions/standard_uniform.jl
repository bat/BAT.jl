# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    StandardUvUniform{T<:Real} <: Distributions.Distribution{Univariate,Continuous}

A standard uniform distribution between zero and one.

Constructor:
```
    StandardUvUniform()
    StandardUvUniform{T<:Real}()
```
"""
struct StandardUvUniform{T<:Real} <: Distributions.Distribution{Univariate,Continuous}
end

StandardUvUniform() = StandardUvUniform{Float64}()

Distributions.Uniform(::StandardUvUniform{T}) where T = Uniform{T}(zero(T), one(T))
Base.convert(::Type{Distributions.Uniform}, d::StandardUvUniform) = Uniform(d)


Base.minimum(::StandardUvUniform{T}) where T = zero(T)
Base.maximum(::StandardUvUniform{T}) where T = one(T)

StatsBase.params(::StandardUvUniform{T}) where T = (zero(T), one(T))
@inline Distributions.partype(::StandardUvUniform{T}) where T = T

Distributions.location(::StandardUvUniform{T}) where T = zero(T)
Distributions.scale(::StandardUvUniform{T}) where T = one(T)

Base.eltype(::Type{StandardUvUniform{T}}) where T = T

Statistics.mean(d::StandardUvUniform{T}) where T = convert(float(T), 1//2)
StatsBase.median(d::StandardUvUniform{T}) where T = mean(d)
StatsBase.mode(d::StandardUvUniform{T}) where T = mean(d)
StatsBase.modes(d::StandardUvUniform{T}) where T = Fill(mode(d), 0)

Statistics.var(d::StandardUvUniform{T}) where T = convert(float(T), 1//12)
StatsBase.std(d::StandardUvUniform{T}) where T = sqrt(var(d))
StatsBase.skewness(d::StandardUvUniform{T}) where T = zero(T)
StatsBase.kurtosis(d::StandardUvUniform{T}) where T = convert(T, -6//5)

StatsBase.entropy(d::StandardUvUniform{T}) where T = zero(T)


function Distributions.logpdf(d::StandardUvUniform{T}, x::U) where {T,U<:Real}
    R = float(promote_type(T,U))
    ifelse(insupport(d, x), R(0), R(-Inf))
end

function Distributions.pdf(d::StandardUvUniform{T}, x::U) where {T,U<:Real}
    R = promote_type(T,U)
    ifelse(insupport(d, x), one(R), zero(R))
end

Distributions.logcdf(d::StandardUvUniform, x::Real) = log(cdf(d, x))

function Distributions.cdf(d::StandardUvUniform{T}, x::U) where {T,U<:Real}
    R = promote_type(T,U)
    ifelse(x < zero(U), zero(R), ifelse(x < one(U), x, one(R)))
end

Distributions.logccdf(d::StandardUvUniform, x::Real) = log(ccdf(d, x))

function Distributions.ccdf(d::StandardUvUniform{T}, x::U) where {T,U<:Real}
    y = cdf(d, x)
    one(y) - y
end

function Distributions.quantile(d::StandardUvUniform{T}, p::U) where {T,U<:Real}
   R = promote_type(T,U)
   convert(float(R), p)
end

function Distributions.cquantile(d::StandardUvUniform{T}, p::U) where {T,U<:Real}
    y = quantile(d, p)
    one(y) - y
end

Distributions.mgf(d::StandardUvUniform, t::Real) = mgf(Uniform(d), t)
Distributions.cf(d::StandardUvUniform, t::Real) = cf(Uniform(d), t)

# Distributions doesn't seem to support gradlogpdf for Uniform()
# # Distributions.gradlogpdf(d::StandardUvUniform, x::Real) = zero(T)

Base.rand(rng::AbstractRNG, d::StandardUvUniform{T}) where T = rand(rng, float(T))

Distributions.truncated(d::StandardUvUniform, l::Real, u::Real) = truncated(Uniform(d), l, u)

function Distributions.product_distribution(dists::AbstractVector{StandardUvUniform{T}}) where T
    StandardMvUniform{T}(length(eachindex(dists)))
end


"""
    StandardMvUniform{T<:Real} <: Distributions.Distribution{Multivariate,Continuous}

A standard `n`-dimensional multivariate uniform distribution, from zero
to one in each dimension.

Constructor:
```
    StandardMvUniform(n::Integer)
    StandardMvUniform{T<:Real}(n::Integer)
```
"""
struct StandardMvUniform{T<:Real} <: Distributions.Distribution{Multivariate,Continuous}
    _dim::Int
end

StandardMvUniform(dim::Integer) = StandardMvUniform{Float64}(dim)

Distributions.Product(d::StandardMvUniform{T}) where T = Distributions.Product(Fill(StandardUvUniform{T}(), length(d)))
Base.convert(::Type{Distributions.Product}, d::StandardMvUniform) = Distributions.Product(d)


function Distributions.insupport(d::StandardMvUniform, x::AbstractVector)
    length(d) == length(eachindex(x)) && all(xi -> insupport(StandardUvUniform(), xi), x)
end

Base.eltype(::Type{StandardMvUniform{T}}) where T = T

Base.length(d::StandardMvUniform{T}) where T = d._dim

@inline function Base.view(d::StandardMvUniform{T}, i::Integer) where T
    Base.@boundscheck Base.checkindex(Bool, Base.OneTo(length(d)), i) || Base.throw_boundserror(d, i)
    StandardUvUniform{T}()
end

function Base.view(d::StandardMvUniform{T}, I::AbstractArray{<:Integer}) where T
    Base.@boundscheck Base.checkindex(Bool, Base.OneTo(length(d)), I) || Base.throw_boundserror(d, I)
    StandardMvUniform{T}(length(eachindex(I)))
end

# Not applicable:
# # StatsBase.params(d::StandardMvUniform)
# # @inline Distributions.partype(d::StandardMvUniform{T})

Statistics.mean(d::StandardMvUniform{T}) where T = Fill(mean(StandardUvUniform()), length(d))
Statistics.var(d::StandardMvUniform{T}) where T = Fill(var(StandardUvUniform()), length(d))
Statistics.cov(d::StandardMvUniform{T}) where T = Diagonal(var(d))

StatsBase.mode(d::StandardMvUniform) = mean(d)
StatsBase.modes(d::StandardMvUniform) = Fill(mean(d), 0)

Distributions.invcov(d::StandardMvUniform{T}) where T = Diagonal(Fill(inv(var(StandardUvUniform())), length(d)))
Distributions.logdetcov(d::StandardMvUniform{T}) where T = logdet(cov(d))

StatsBase.entropy(d::StandardMvUniform{T}) where T = zero(T)


function Distributions.logpdf(d::StandardMvUniform{T}, x::AbstractVector{U}) where {T,U<:Real}
    R = float(promote_type(T,U))
    ifelse(insupport(d, x), R(0), R(-Inf))
end

function Distributions.pdf(d::StandardMvUniform{T}, x::AbstractVector{U}) where {T,U<:Real}
    R = promote_type(T,U)
    ifelse(insupport(d, x), one(R), zero(R))
end

# Not useful:
# # Distributions.sqmahal(d::StandardMvUniform, x::AbstractVector{<:Real})
# # Distributions.sqmahal!(r::AbstractVector{<:Real}, d::StandardMvUniform, x::AbstractMatrix{<:Real})

function Distributions.gradlogpdf(d::StandardMvUniform{T}, x::AbstractVector{U}) where {T,U<:Real}
    FillArrays.Zeros(length(d))
end

function Distributions._rand!(rng::AbstractRNG, d::StandardMvUniform, A::AbstractVector{T}) where {T<:Real}
    broadcast!(x -> rand(rng, T), A, A)
end

function Distributions._rand!(rng::AbstractRNG, d::StandardMvUniform, A::AbstractMatrix{T}) where {T<:Real}
    broadcast!(x -> rand(rng, T), A, A)
end
