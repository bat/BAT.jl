# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Compat
using PDMats


@compat abstract type AbstractProposalFunction end


Base.rand!(q::AbstractProposalFunction, new_params::ParamValues, params::ParamValues) =
    rand!(Base.GLOBAL_RNG, q, new_params, params)


immutable StudentTProposalFunction{T<:Real,Cov<:AbstractPDMat} <: AbstractProposalFunction
    ν::T
    Σ::Cov
end


function Base.rand!(rng::AbstractRNG, q::StudentTProposalFunction, new_params::ParamValues, params::ParamValues)
    #rand!(rng, dist, new_params)
    #new_params .+= params
    #ndims = ...
    randn!(rng, new_params)
    new_params .= q.Σ.chol * new_params
    rnd_chi2 = BCMath::Random::Chi2(rng, q.ν);
    scale = sqrt(dof / rnd_chi2);

    new_params .= new_params .* scale .+ params
end


#=

# Add gaussian proposal function, based on Distributions.MvNormal:

_rand!(d::MvNormal, x::VecOrMat) = add!(unwhiten!(d.Σ, randn!(x)), d.μ)


# Use from Distributions.GenericMvTDist (instead of defining ν and Σ in StudentTProposalFunction):

immutable GenericMvTDist{T<:Real, Cov<:AbstractPDMat} <: AbstractMvTDist
    df::T # non-integer degrees of freedom allowed
    dim::Int
    zeromean::Bool
    μ::Vector{T}
    Σ::Cov

    function (::Type{GenericMvTDist{T,Cov}}){T,Cov}(df::T, dim::Int, zmean::Bool, μ::Vector{T}, Σ::AbstractPDMat{T})
      df > zero(df) || error("df must be positive")
      new{T,Cov}(df, dim, zmean, μ, Σ)
    end
end


function _rand!{T<:Real}(d::GenericMvTDist, x::AbstractVector{T})
    chisqd = Chisq(d.df)
    y = sqrt(rand(chisqd)/(d.df))
    unwhiten!(d.Σ, randn!(x))
    broadcast!(/, x, x, y)
    if !d.zeromean
        broadcast!(+, x, x, d.μ)
    end
    x
end

=#
