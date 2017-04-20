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
