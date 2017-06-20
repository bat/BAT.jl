# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Distributions


abstract type AbstractProposalFunction end


export ProposalFunction

struct ProposalFunction{D<:Distribution,SamplerF} <: AbstractProposalFunction
    d::D
    sampler_f::SamplerF

    function ProposalFunction{D,SamplerF}(d::D, sampler_f::SamplerF) where {D<:Distribution,SamplerF}
        issymmetric_at_origin(d) || throw(ArgumentError("Distribution $d must be symmetric at origin"))
        new{D,SamplerF}(d, sampler_f)
    end

end

ProposalFunction{D<:Distribution,SamplerF}(d::D, sampler_f::SamplerF) = ProposalFunction{D,SamplerF}(d, sampler_f)

ProposalFunction(d::Distribution) = ProposalFunction(d, bat_sampler)

Distributions.sampler(q::ProposalFunction) = q.sampler_f(q.d)
