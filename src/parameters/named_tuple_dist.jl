# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ConstValueDistribution{VF<:VariateForm,T} <: Distribution{VF,Continuous}
    value::T
end

ConstValueDistribution(x::T) where {T<:Real} = ConstValueDistribution{Univariate,T}(x)
ConstValueDistribution(x::T) where {T<:AbstractVector{<:Real}} = ConstValueDistribution{Multivariate,T}(x)
ConstValueDistribution(x::T) where {T<:AbstractMatrix{<:Real}} = ConstValueDistribution{Matrixvariate,T}(x)

Distributions.pdf(d::ConstValueDistribution{Univariate}, x::Real) = d.value == x ? float(eltype(d))(Inf) : float(eltype(d))(0)
Distributions._logpdf(d::ConstValueDistribution, x::AbstractArray{<:Real}) = d.value == x ? float(eltype(d))(Inf) : float(eltype(d))(0)

Distributions.cdf(d::ConstValueDistribution{Univariate}, x::Real) = d.value <= x ? Float32(1) : Float32(0)
Distributions.quantile(d::ConstValueDistribution{Univariate}, q::Real) = d.value # Sensible?
Distributions.minimum(d::ConstValueDistribution{Univariate}) = x.value
Distributions.maximum(d::ConstValueDistribution{Univariate}) = x.value
Distributions.insupport(d::ConstValueDistribution{Univariate}, x::Real) = x == d.value

Base.size(d::ConstValueDistribution) = size(d.value)
Base.length(d::ConstValueDistribution) = prod(size(d))
Base.eltype(d::ConstValueDistribution) = eltype(d.value)

Random.rand(rng::AbstractRNG, d::ConstValueDistribution) = d.value

function Random._rand!(rng::AbstractRNG, d::ConstValueDistribution, params::AbstractArray{<:Real})
    copyto!(params, d.value)
end


_np_valshape(d::Distribution{Univariate,Continuous}) = ScalarShape{Real}()
_np_bounds(d::Distribution{Univariate,Continuous}) = HyperRectBounds([quantile(d, 0f0)], [quantile(d, 1f0)], reflective_bounds)

_np_valshape(d::Distribution{Multivariate,Continuous}) = ArrayShape{Real}(size(d)...)
_np_bounds(d::Distribution{Multivariate,Continuous}) = HyperRectBounds(fill(Float32(-Inf), length(d)), fill(Float32(+Inf), length(d)), hard_bounds)

_np_bounds(d::ConstValueDistribution) = HyperRectBounds(Vector{eltype(d)}(), Vector{eltype(d)}(), hard_bounds)
_np_bounds(d::Product{Continuous}) = HyperRectBounds(quantile.(d.v, 0f0), quantile.(d.v, 1f0), reflective_bounds)

_np_valshape(d::Distribution{Matrixvariate,Continuous}) = ArrayShape{Real}(size(d)...)

_np_dist_shape_bounds(d::Distribution) = (d, _np_valshape(d), _np_bounds(d))

_np_distribution(s::ConstValueShape) = ConstValueDistribution(s.value)
_np_bounds(s::ConstValueShape) = HyperRectBounds(Float64[], Float64[], hard_bounds)
_np_dist_shape_bounds(s::ConstValueShape) = (_np_distribution(s), s, _np_bounds(s))

_np_dist_shape_bounds(s::AbstractInterval) = _np_dist_shape_bounds(Uniform(minimum(s), maximum(s)))
_np_dist_shape_bounds(xs::AbstractVector{<:AbstractInterval}) = _np_dist_shape_bounds(Product((s -> Uniform(minimum(s), maximum(s))).(xs)))
_np_dist_shape_bounds(xs::AbstractVector{<:Distribution}) = _np_dist_shape_bounds(Product(xs))
_np_dist_shape_bounds(x::Number) = _np_dist_shape_bounds(ConstValueShape(x))
_np_dist_shape_bounds(x::AbstractArray{<:Number}) = _np_dist_shape_bounds(ConstValueShape(x))



"""
    NamedTupleDist <: MultivariateDistribution
    NamedTupleDist <: MultivariateDistribution

A distribution with `NamedTuple`-typed variates.

Can be used to describe the distribution of each parameter in a set of
named parameters. If the distribution is used as a Bayesian prior,
the `NamedTupleDist` then specifies the prior on each named parameter.

A `NamedTupleDist` implies a `NamedTupleShape`:

    valshape(d::NamedTupleDist) isa NamedTupleShape
"""
struct NamedTupleDist{
    names,
    DT <: (NTuple{N,Distribution} where N),
    AT <: (NTuple{N,ValueShapes.ValueAccessor} where N),
    BT <: (NTuple{N,AbstractParamBounds} where N)
} <: Distribution{Multivariate,Continuous}
    _distributions::NamedTuple{names,DT}
    _shapes::NamedTupleShape{names,AT}
    _bounds::NamedTuple{names,BT}
end 

export NamedTupleDist


function NamedTupleDist(param_priors::NamedTuple{names}) where {names}
    dsb = map(_np_dist_shape_bounds, param_priors)
    NamedTupleDist(
        map(x -> x[1], dsb),
        NamedTupleShape(map(x -> x[2], dsb)),
        map(x -> x[3], dsb)
    )
end

@inline NamedTupleDist(;named_priors...) = NamedTupleDist(values(named_priors))



@inline _distributions(p::NamedTupleDist) = getfield(p, :_distributions)
@inline _shapes(p::NamedTupleDist) = getfield(p, :_shapes)
@inline _bounds(p::NamedTupleDist) = getfield(p, :_bounds)


@inline Base.keys(p::NamedTupleDist) = keys(_distributions(p))

@inline Base.values(p::NamedTupleDist) = values(_distributions(p))

@inline function Base.getproperty(p::NamedTupleDist, s::Symbol)
    # Need to include internal fields of NamedTupleShape to make Zygote happy:
    if s == :_distributions
        getfield(p, :_distributions)
    elseif s == :_shapes
        getfield(p, :_shapes)
    elseif s == :_bounds
        getfield(p, :_bounds)
    else
        getproperty(_distributions(p), s)
    end
end

@inline Base.propertynames(p::NamedTupleDist) = propertynames(_distributions(p))


ValueShapes.valshape(p::NamedTupleDist) = _shapes(p)


param_bounds(p::NamedTupleDist) = vcat(map(_np_bounds, values(p))...)


_np_length(d::Distribution) = length(d)
_np_length(d::ConstValueDistribution) = 0

Base.length(p::NamedTupleDist) = sum(_np_length, values(p))


function _np_logpdf(
    dist::ConstValueDistribution,
    acc::ValueShapes.ValueAccessor{<:ConstValueShape},
    params::AbstractVector{<:Real}
)
    float(zero(eltype(params)))
end

function _np_logpdf(
    dist::Distribution,
    acc::ValueShapes.ValueAccessor,
    params::AbstractVector{<:Real}
)
    logpdf(dist, float(params[acc]))
end

function Distributions.logpdf(p::NamedTupleDist, params::AbstractVector{<:Real})
    distributions = values(p)
    accessors = values(valshape(p))
    sum(map((dist, acc) -> _np_logpdf(dist, acc, params), distributions, accessors))
end


# ConstValueDistribution has no dof, so NamedTupleDist logpdf contribution must be zero:
_np_logpdf(dist::ConstValueDistribution, params::Any) = zero(Float32)

_np_logpdf(dist::Distribution, params::Any) = logpdf(dist, params)

function Distributions.logpdf(p::NamedTupleDist{names}, params::NamedTuple{names}) where names
    distributions = values(p)
    parvalues = values(params)
    sum(map((dist, p) -> _np_logpdf(dist, p), distributions, parvalues))
end


function _np_rand!(
    rng::AbstractRNG, dist::ConstValueDistribution,
    acc::ValueShapes.ValueAccessor{<:ConstValueShape},
    params::AbstractVector{<:Real}
)
    nothing
end

function _np_rand!(
    rng::AbstractRNG, dist::Distribution,
    acc::ValueShapes.ValueAccessor,
    params::AbstractVector{<:Real}
)
    rand!(rng, dist, view(params, acc))
    nothing
end

function Distributions._rand!(rng::AbstractRNG, p::NamedTupleDist, params::AbstractVector{<:Real})
    distributions = values(p)
    accessors = values(valshape(p))
    map((dist, acc) -> _np_rand!(rng, dist, acc, params), distributions, accessors)
    params
end

#Random.rand(rng::AbstractRNG, p::NamedTupleDist) = rand!(rng, p, Vector{Float64}(undef, length(p)))


function _np_var_or_cov!(A_cov::AbstractArray{<:Real,0}, dist::Distribution{Univariate})
    A_cov[] = var(dist)
    nothing
end

function _np_var_or_cov!(A_cov::AbstractArray{<:Real,2}, dist::Distribution{Multivariate})
    A_cov[:, :] = cov(dist)
    nothing
end

function _np_cov!(
    dist::ConstValueDistribution,
    acc::ValueShapes.ValueAccessor{<:ConstValueShape},
    A_cov::AbstractMatrix{<:Real}
)
    nothing
end

function _np_cov!(
    dist::Distribution,
    acc::ValueShapes.ValueAccessor,
    A_cov::AbstractMatrix{<:Real}
)
    _np_var_or_cov!(view(A_cov, acc, acc), dist)
    nothing
end

function Statistics.cov(p::NamedTupleDist)
    let n = length(p), A_cov = zeros(n, n)
        distributions = values(p)
        accessors = values(valshape(p))
        map((dist, acc) -> _np_cov!(dist, acc, A_cov), distributions, accessors)
        A_cov
    end
end
