# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct BATDistMeasure{D<:Distribution} <: BATMeasure
    dist::D
end

BATMeasure(d::ContinuousDistribution) = BATDistMeasure(d)

Distributions.Distribution(m::BATDistMeasure) = m.dist
Base.convert(::Type{Distribution}, d::BATDistMeasure) = Distribution(d)

Base.:(==)(a::BATDistMeasure, b::BATDistMeasure) = a.dist == b.dist

MeasureBase.getdof(m::BATDistMeasure) = eff_totalndof(m.dist)

MeasureBase.rootmeasure(::BATDistMeasure{<:Distribution{Univariate,Continuous}}) = MeasureBase.LebesgueBase()
MeasureBase.rootmeasure(m::BATDistMeasure{<:Distribution{Multivariate,Continuous}}) = MeasureBase.LebesgueBase() ^ size(m.dist)

MeasureBase.massof(::BATDistMeasure) = 1.0f0


function DensityInterface.logdensityof(m::BATDistMeasure{<:Distribution{Univariate,Continuous}}, v::Real)
    d = m.dist
    logd = logpdf(d, v)
    R = typeof(logd)
    # ToDo: Move these workarounds somewhere else? Still necessary at all?
    if isnan(logd)
        if isinf(v)
            # Weibull yields NaN logpdf at infinity (Distributions.jl issue #1197), possibly others too,
            # so force to -Inf (there should never be any probability mass at infinity):
            convert(R, -Inf)
        elseif v ≈ minimum(d)
            # Weibull yields NaN logpdf at 0 (Distributions.jl issue #1197), possibly others too,
            # so move an epsilon away from minimum:
            convert(R, logpdf(d, minimum(d) + eps(typeof(v))))
        elseif v ≈ maximum(d)
            # Likewise at maxiumum:
            convert(R, logpdf(d, maximum(d) - eps(typeof(v))))
        else
            logd
        end
    else
        logd
    end
end

function DensityInterface.logdensityof(m::BATDistMeasure{<:Distribution{Univariate,Continuous}}, v)
    throw(ArgumentError("logdensityof not defined for $(nameof(typeof(m))) and $(nameof(typeof(v)))"))
end

DensityInterface.logdensityof(m::BATDistMeasure, v) = logdensityof(m.dist, v)


ValueShapes.varshape(m::BATDistMeasure) = varshape(m.dist)

ValueShapes.unshaped(m::BATDistMeasure) = BATDistMeasure(unshaped(m.dist))

(shape::AbstractValueShape)(m::BATDistMeasure) = BATDistMeasure(shape(m.dist))


function Random.rand(gen::GenContext, m::BATDistMeasure)
    gen_adapt(gen, rand(get_rng(gen), m.dist))
end

_reshape_rand_n_output(x::Any) = x
x =_reshape_rand_n_output(x::AbstractMatrix) = nestedview(x)
_reshape_rand_n_output(x::AbstractArray{<:AbstractArray}) = ArrayOfSimilarArrays(x)
_reshape_rand_n_output(x::ArrayOfSimilarArrays) = x

import Random.rand
Base.@deprecate rand(rng::AbstractRNG, m::BATDistMeasure, dims::Dims) rand(rng, m^dims)
Base.@deprecate rand(rng::AbstractRNG, m::BATDistMeasure, dim::Integer, dims::Integer...) rand(rng, m^(dim, dims...))

@inline supports_rand(::BATDistMeasure) = true

Statistics.mean(m::BATDistMeasure{<:MultivariateDistribution}) = mean(m.dist)
Statistics.var(m::BATDistMeasure{<:MultivariateDistribution}) = var(m.dist)
Statistics.cov(m::BATDistMeasure{<:MultivariateDistribution}) = cov(m.dist)


measure_support(m::BATDistMeasure) = dist_support(m.dist)

is_std_mvnormal(m::BATDistMeasure) = is_std_mvnormal(m.dist)


dist_support(d::Distribution) = UnknownVarBounds()

dist_support(d::StandardUvUniform) = UnitInterval()
dist_support(d::StandardUvNormal) = RealNumbers()
dist_support(d::StandardMvUniform) = UnitCube(prod(size(d)))
dist_support(::StandardMvNormal) = FullSpace()

dist_support(d::Distribution{Univariate,Continuous}) = ClosedInterval(minimum(d), maximum(d))
dist_support(d::Normal) = RealNumbers()
dist_support(d::AbstractMvNormal) = FullSpace()

dist_support(d::ReshapedDist) = dist_support(unshaped(d))

dist_support(d::Product{<:Continuous,<:Distribution{Univariate}}) = Rectangle(map(dist_support, d.v))


is_std_mvnormal(::Distribution) = false
is_std_mvnormal(::StandardMvNormal) = true
is_std_mvnormal(d::MvNormal) = mean(d) ≈ Zeros(length(d)) && cov(d) ≈ I(length(d))
