# This file is a part of BAT.jl, licensed under the MIT License (MIT).

_valshape(d::Distribution{Univariate,Continuous}) = ScalarShape{Real}()
_bounds(d::Distribution{Univariate,Continuous}) = HyperRectBounds([quantile(d, 0f0)], [quantile(d, 1f0)], reflective_bounds)

_valshape(d::Distribution{Multivariate,Continuous}) = ArrayShape{Real}(size(d)...)
_bounds(d::Distribution{Multivariate,Continuous}) = HyperRectBounds(fill(Float32(-Inf), length(d)), fill(Float32(+Inf), length(d)), hard_bounds)

_bounds(d::Product{Continuous}) = HyperRectBounds(quantile.(d.v, 0f0), quantile.(d.v, 1f0), reflective_bounds)

_valshape(d::Distribution{Matrixvariate,Continuous}) = ArrayShape{Real}(size(d)...)

_dist_shape_bounds(d::Distribution) = (d, _valshape(d), _bounds(d))

_distribution(s::ConstValueShape) = Product(Uniform{Float64}[])
_bounds(s::ConstValueShape) = HyperRectBounds(Float64[], Float64[], hard_bounds)
_dist_shape_bounds(s::ConstValueShape) = (_distribution(s), s, _bounds(s))

_dist_shape_bounds(s::AbstractInterval) = _dist_shape_bounds(Uniform(minimum(s), maximum(s)))
_dist_shape_bounds(xs::AbstractVector{<:AbstractInterval}) = _dist_shape_bounds(Product((s -> Uniform(minimum(s), maximum(s))).(xs)))
_dist_shape_bounds(x::Number) = _dist_shape_bounds(ConstValueShape(x))
_dist_shape_bounds(x::AbstractArray{<:Number}) = _dist_shape_bounds(ConstValueShape(x))


struct NamedPrior{
    names,
    DT <: (NTuple{N,Distribution} where N),
    AT <: (NTuple{N,ShapesOfVariables.VariableDataAccessor} where N),
    BT <: (NTuple{N,AbstractParamBounds} where N)
} <: Distribution{Multivariate,Continuous}
    _distributions::NamedTuple{names,DT}
    _shapes::VarShapes{names,AT}
    _bounds::NamedTuple{names,BT}
end 

export NamedPrior

function NamedPrior(param_priors::NamedTuple{names}) where {names}
    dsb = map(_dist_shape_bounds, param_priors)
    NamedPrior(
        map(x -> x[1], dsb),
        VarShapes(map(x -> x[2], dsb)),
        map(x -> x[3], dsb)
    )
end

@inline NamedPrior(;named_priors...) = NamedPrior(named_priors.data)



@inline _distributions(p::NamedPrior) = getfield(p, :_distributions)
@inline _shapes(p::NamedPrior) = getfield(p, :_shapes)
@inline _bounds(p::NamedPrior) = getfield(p, :_bounds)


@inline Base.keys(p::NamedPrior) = keys(_distributions(p))

@inline Base.values(p::NamedPrior) = values(_distributions(p))

@inline Base.getproperty(p::NamedPrior, s::Symbol) = getproperty(_distributions(p), s)

@inline Base.propertynames(p::NamedPrior) = propertynames(_distributions(p))


VarShapes(p::NamedPrior) = _shapes(p)
Base.convert(::Type{VarShapes}, p) = VarShapes(p)

VarShapes{names}(p::NamedPrior{names}) where {names} = _shapes(p)
Base.convert(::Type{VarShapes{names}}, p) where {names} = VarShapes(p)

param_bounds(p::NamedPrior) = vcat(map(_bounds, values(p))...)


Base.length(p::NamedPrior) = sum(length, values(p))

# ToDo: Improve:
Distributions.sampler(p::NamedPrior) = p


function _np_logpdf(
    dist::Distribution,
    acc::ShapesOfVariables.VariableDataAccessor{<:ConstValueShape},
    params::AbstractVector{<:Real}
)
    float(zero(eltype(params)))
end

function _np_logpdf(
    dist::Distribution,
    acc::ShapesOfVariables.VariableDataAccessor,
    params::AbstractVector{<:Real}
)
    logpdf(dist, float(params[acc]))
end

function Distributions.logpdf(p::NamedPrior, params::AbstractVector{<:Real})
    distributions = values(p)
    accessors = values(VarShapes(p))
    sum(map((dist, acc) -> _np_logpdf(dist, acc, params), distributions, accessors))
end


function _np_rand!(
    rng::AbstractRNG, dist::Distribution,
    acc::ShapesOfVariables.VariableDataAccessor{<:ConstValueShape},
    params::AbstractVector{<:Real}
)
    length(dist) == 0 || throw(ArgumentError("Expected a distribution 0 degrees of freedom"))
    nothing
end

function _np_rand!(
    rng::AbstractRNG, dist::Distribution,
    acc::ShapesOfVariables.VariableDataAccessor,
    params::AbstractVector{<:Real}
)
    rand!(rng, dist, view(params, acc))
    nothing
end

function Random.rand!(rng::AbstractRNG, p::NamedPrior, params::AbstractVector{<:Real})
    distributions = values(p)
    accessors = values(VarShapes(p))
    map((dist, acc) -> _np_rand!(rng, dist, acc, params), distributions, accessors)
    params
end

Random.rand(rng::AbstractRNG, p::NamedPrior) = rand!(rng, p, Vector{Float64}(undef, length(p)))


function _np_var_or_cov!(A_cov::AbstractArray{<:Real,0}, dist::Distribution{Univariate})
    A_cov[] = var(dist)
    nothing
end

function _np_var_or_cov!(A_cov::AbstractArray{<:Real,2}, dist::Distribution{Multivariate})
    A_cov[:, :] = cov(dist)
    nothing
end

function _np_cov!(
    dist::Distribution,
    acc::ShapesOfVariables.VariableDataAccessor{<:ConstValueShape},
    A_cov::AbstractMatrix{<:Real}
)
    length(dist) == 0 || throw(ArgumentError("Expected a distribution 0 degrees of freedom"))
    nothing
end

function _np_cov!(
    dist::Distribution,
    acc::ShapesOfVariables.VariableDataAccessor,
    A_cov::AbstractMatrix{<:Real}
)
    _np_var_or_cov!(view(A_cov, acc, acc), dist)
    nothing
end

function Statistics.cov(p::NamedPrior) 
    let n = length(p), A_cov = zeros(n, n)
        distributions = values(p)
        accessors = values(VarShapes(p))
        map((dist, acc) -> _np_cov!(dist, acc, A_cov), distributions, accessors)
        A_cov
    end
end
