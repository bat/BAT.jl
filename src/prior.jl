# This file is a part of BAT.jl, licensed under the MIT License (MIT).



abstract type AbstractPrior{HasBounds} <: AbstractDensityFunction{true,HasBounds,false} end

# XXXXXX !!!!! Alternative:
# const AbstractPrior{HasBounds} = AbstractDensityFunction{true,HasBounds,false}


const AbstractBoundedPrior = AbstractPrior{true}

const AbstractUnboundedPrior = AbstractPrior{false}


param_prior(density::AbstractDensityFunction{<:Any,<:Any,false}) = NoPrior()



struct NoPrior{T<:Real} <: AbstractUnboundedPrior  XXXX check
    nparams::Int
end
b
export NoPrior


nparams(density::NoPrior) = density.nparams

function density_logval(
    density::NoPrior,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    one(eltype(params))
end

function density_logval!(
    r::AbstractArray{<:Real},
    density::NoPrior,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    size(params, 2) != length(r) && throw(ArgumentError("Number of parameter vectors doesn't match length of result vector"))
    fill!(r, one(eltype(r)))
end



abstract type AbstractPriorDistribution <: AbstractUnboundedPrior



struct PriorDistribution{D<:Distribution{Multivariate}} <: AbstractPriorDistribution
    distribution::D
end


Base.parent(p::PriorDistribution) = p.distribution


nparams(density::PriorDistribution) = length(parent(density))

function density_logval(
    density::PriorDistribution,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    Distributions.logpdf(parent(density), params)
end

function density_logval!(
    r::AbstractArray{<:Real},
    density::PriorDistribution,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid number of parameters"))
    size(params, 2) != length(r) && throw(ArgumentError("Number of parameter vectors doesn't match length of result vector"))
    Distributions.logpdf!(r, parent(density), params)
end


Distributions.sampler(p::PriorDistribution) = bat_sampler(parent(p))

rand!(rng::AbstractRNG, p::PriorDistribution, p::VecOrMat) = rand!(rng, sampler(p), p)
rand(rng::AbstractRNG, p::PriorDistribution, n::Integer) = rand(rng, sampler(p), n)


# TODO: XXXXX BoundedPrior, like BoundedDensity
