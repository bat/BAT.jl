# This file is a part of BAT.jl, licensed under the MIT License (MIT).



abstract type AbstractPrior{HasBounds} <: AbstractDensityFunction{true,HasBounds,false} end

# XXXXXX !!!!! Alternative:
# const AbstractPrior{HasBounds} = AbstractDensityFunction{true,HasBounds,false}


const AbstractBoundedPrior = AbstractPrior{true}

const AbstractUnboundedPrior = AbstractPrior{false}


struct NoPrior{T<:Real} <: AbstractUnboundedPrior  XXXX check
    nparams::Int
end
b
export NoPrior


nparams(density::NoPrior) = density.nparams

function unsafe_density_logval(
    density::NoPrior,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    one(eltype(params))
end

function unsafe_density_logval!(
    r::AbstractArray{<:Real},
    density::NoPrior,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext
)
    fill!(r, one(eltype(r)))
end



abstract type AbstractPriorDistribution <: AbstractUnboundedPrior



struct PriorDistribution{D<:Distribution{Multivariate}} <: AbstractPriorDistribution
    distribution::D
end


Base.parent(p::PriorDistribution) = p.distribution


nparams(density::PriorDistribution) = length(parent(density))

function unsafe_density_logval(
    density::PriorDistribution,
    params::AbstractVector{<:Real},
    exec_context::ExecContext = ExecContext()
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid length of parameter vector"))
    Distributions.logpdf(parent(density), params)
end

function unsafe_density_logval!(
    r::AbstractArray{<:Real},
    density::PriorDistribution,
    params::AbstractMatrix{<:Real},
    exec_context::ExecContext
)
    size(params, 1) != nparams(density) && throw(ArgumentError("Invalid length of parameter vector"))
    size(params, 2) != length(r) && throw(ArgumentError("Number of parameter vectors doesn't match length of result vector"))
    Distributions.logpdf!(r, parent(density), params)
end


Distributions.sampler(p::PriorDistribution) = bat_sampler(parent(p))

rand!(rng::AbstractRNG, p::PriorDistribution, p::VecOrMat) = rand!(rng, sampler(p), p)
rand(rng::AbstractRNG, p::PriorDistribution, n::Integer) = rand(rng, sampler(p), n)


# TODO: XXXXX BoundedPrior, like BoundedDensity
