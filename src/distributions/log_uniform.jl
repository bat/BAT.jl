# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BAT.LogUniform{T<:Real} <: Distributions.Distribution{Univariate,Continuous}

*Experimental feature, not part of stable public API.*

The log-uniform distribution (reciprocal distribution) over an interval ``[a, b]``.

Constructors:

* ```LogUniform(a::Real, b::Real)```

See also [Reciprocal distribution on Wikipedia](https://en.wikipedia.org/wiki/Reciprocal_distribution).
"""
struct LogUniform{T<:Real} <: Distributions.Distribution{Univariate,Continuous}
    a::T
    b::T
    _normf::T
end

function LogUniform{T}(a::Real, b::Real) where {T<:Real}
    @argcheck 0 < T(a) < T(b)
    LogUniform{T}(a, b, inv(log(b / a)))
end

function LogUniform(a::Real, b::Real)
    T = promote_type(typeof(a), typeof(b))
    LogUniform{T}(a, b)
end

LogUniform() = LogUniform{Float64}(0, 1)


Base.minimum(d::LogUniform{T}) where T = d.a
Base.maximum(d::LogUniform{T}) where T = d.b

StatsBase.params(d::LogUniform{T}) where T = (d.a, d.b)
@inline Distributions.partype(::LogUniform{T}) where T = T

Distributions.location(d::LogUniform) = d.a
Distributions.scale(d::LogUniform) = d.b - d.a


Base.eltype(::Type{LogUniform{T}}) where T = T

Statistics.mean(d::LogUniform{T}) where T = (d.b - d.a) * d._normf
StatsBase.median(d::LogUniform{T}) where T = quantile(d, 1//2)
StatsBase.mode(d::LogUniform{T}) where T = d.a
StatsBase.modes(d::LogUniform{T}) where T = Fill(mode(d), 0)

Statistics.var(d::LogUniform{T}) where T = (d.b^2 - d.a^2)/2 * d._normf - mean(d)^2
StatsBase.std(d::LogUniform{T}) where T = sqrt(var(d))
# StatsBase.skewness(d::LogUniform{T}) where T = ...
# StatsBase.kurtosis(d::LogUniform{T}) where T = ...

# StatsBase.entropy(d::LogUniform{T}) where T = ...


function Distributions.logpdf(d::LogUniform{T}, x::U) where {T,U<:Real}
    log(pdf(d, x))
end

function Distributions.pdf(d::LogUniform{T}, x::U) where {T,U<:Real}
    R = promote_type(T,U)
    y = inv(x) * d._normf
    ifelse(insupport(d, x), R(y), zero(R))
end

Distributions.logcdf(d::LogUniform, x::Real) = log(cdf(d, x))

function Distributions.cdf(d::LogUniform{T}, x::U) where {T,U<:Real}
    R = promote_type(T,U)
    p = log(x / d.a) * d._normf
    ifelse(insupport(d, x), R(p), zero(R))
end

Distributions.logccdf(d::LogUniform, x::Real) = log(ccdf(d, x))

function Distributions.ccdf(d::LogUniform{T}, x::U) where {T,U<:Real}
    y = cdf(d, x)
    one(y) - y
end

function Distributions.quantile(d::LogUniform{T}, p::U) where {T,U<:Real}
   R = promote_type(T,U)
   x = exp(p / d._normf) * d.a
   convert(float(R), x)
end

function Distributions.cquantile(d::LogUniform{T}, p::U) where {T,U<:Real}
    y = quantile(d, p)
    one(y) - y
end

# Distributions.mgf(d::LogUniform, t::Real) = ...
# Distributions.cf(d::LogUniform, t::Real) = ...

# # Distributions.gradlogpdf(d::LogUniform, x::Real) = ...

# Implemented implicitly via quantile:
# Base.rand(rng::AbstractRNG, d::LogUniform)

Distributions.truncated(d::LogUniform, l::Real, u::Real) = LogUniform(promote(max(l, d.a), min(u, d.b))...)
Distributions.truncated(d::LogUniform, l::Integer, u::Integer) = LogUniform(promote(max(l, d.a), min(u, d.b))...)
