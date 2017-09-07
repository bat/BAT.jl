# This file is a part of BAT.jl, licensed under the MIT License (MIT).




abstract type AbstractMCMCState end


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


Base.length(sample::MCMCSample) = length(sample.params)

Base.similar(sample::MCMCSample{P,T,W}) where {P,T,W} = MCMCSample{P,T,W}(similar(sample.params), 0, 0, 0)

function Base.copy!(dest::MCMCSample, src::MCMCSample) 
    copy!(dest.params, src.params)
    dest.log_value = src.log_value
    dest.weight = src.weight
    dest
end





@enum MCMChainState UNCONVERGED=0 CONVERGED=1
export MCMChainState # Better name for this?
export UNCONVERGED
export CONVERGED


struct MCMCChainInfo
    id::Int
    cycle::Int
    state::MCMChainState
end

export MCMCChainInfo

MCMCChainInfo() = MCMCChainInfo(0, 0, UNCONVERGED)



struct MCMCChainStats{T<:Real,P<:Real}
    param_stats::BasicMvStatistics{P,FrequencyWeights}
    logtf_stats::BasicUvStatistics{T,FrequencyWeights}
    mode::Vector{P}
end

export MCMCChainStats



struct MCMCChain{
    A<:MCMCAlgorithm,
    T<:AbstractTargetSubject,
    S<:AbstractMCMCState,
    R<:AbstractRNG
}
    algorithm::A
    target::T
    state::S
    nsamples::Int64  # -> state?
    rng::R
    info::MCMCChainInfo
end

export MCMCChain





"""
    mcmc_step(state::AbstractMCMCState, rng::AbstractRNG, exec_context::ExecContext = ExecContext())
    mcmc_step(states::AbstractVector{<:AbstractMCMCState}, rng::AbstractRNG, exec_context::ExecContext = ExecContext()) where {P,R}
"""
function  mcmc_step end
export mcmc_step


"""
    exec_context(state::AbstractMCMCState)
"""
function exec_context end
export exec_context
