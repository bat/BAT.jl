# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCState end


sample_available(state::AbstractMCMCState) = sample_available(state, Val(:complete))

current_sample(state::AbstractMCMCState) = current_sample(state, Val(:complete))



abstract type MCMCAlgorithm{S<:AbstractMCMCState} end


abstract type AbstractMCMCSample end
export AbstractMCMCSample



mutable struct MCMCSample{
    P<:Real,
    T<:Real,
    W<:Real
} <: AbstractMCMCSample
    params::Vector{P}
    log_value::T
    weight::W
end

export MCMCSample


Base.length(s::MCMCSample) = length(s.params)

Base.similar(s::MCMCSample{P,T,W}) where {P,T,W} =
    MCMCSample{P,T,W}(oob(s.params), convert(T, NaN), zero(W))

import Base.==
==(A::MCMCSample, B::MCMCSample) =
    A.params == B.params && A.log_value == B.log_value && A.weight == B.weight


function Base.copy!(dest::MCMCSample, src::MCMCSample) 
    copy!(dest.params, src.params)
    dest.log_value = src.log_value
    dest.weight = src.weight
    dest
end


nparams(s::MCMCSample) = length(s)



struct MCMCChainInfo
    id::Int
    cycle::Int
    tuned::Bool
    converged::Bool
end

export MCMCChainInfo

MCMCChainInfo(id::Int, cycle::Int = 0) = MCMCChainInfo(id, cycle, false, false)


next_cycle(info::MCMCChainInfo) =
    MCMCChainInfo(info.id, info.cycle + 1, info.tuned, info.converged)

set_tuned(info::MCMCChainInfo, value::Bool) =
    MCMCChainInfo(info.id, info.cycle, value, info.converged)

set_converged(info::MCMCChainInfo, value::Bool) =
    MCMCChainInfo(info.id, info.cycle, info.tuned, value)



mutable struct MCMCChain{
    A<:MCMCAlgorithm,
    T<:AbstractTargetSubject,
    S<:AbstractMCMCState
}
    algorithm::A
    target::T
    state::S
    info::MCMCChainInfo
end

export MCMCChain


nparams(chain::MCMCChain) = nparams(chain.target)
