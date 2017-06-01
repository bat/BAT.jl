# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Compat
using Distributions


@compat abstract type AbstractProposalFunction end


export ProposalFunction

immutable ProposalFunction{D<:Distribution,SamplerF} <: AbstractProposalFunction
    d::D
    sampler_f::SamplerF
end


ProposalFunction(d::Distribution) = ProposalFunction(d, bat_sampler)

Distributions.sampler(q::ProposalFunction) = q.sampler_f(q.d)
