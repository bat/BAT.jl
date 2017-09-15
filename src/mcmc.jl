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



struct MCMCChainStats{L<:Real,P<:Real}
    param_stats::BasicMvStatistics{P,FrequencyWeights}
    logtf_stats::BasicUvStatistics{L,FrequencyWeights}
    mode::Vector{P}

    function MCMCChainStats{L,P}(m::Integer) where {L<:Real,P<:Real}
        param_stats = BasicMvStatistics{P,FrequencyWeights}(m)
        logtf_stats = BasicUvStatistics{L,FrequencyWeights}()
        mode = Vector{P}(size(param_stats.mean, 1))

        new{L,P}(
            BasicMvStatistics{P,FrequencyWeights}(m),
            BasicUvStatistics{L,FrequencyWeights}(),
            fill(oob(P), m)
        )
    end
end

export MCMCChainStats



struct MCMCChain{
    A<:MCMCAlgorithm,
    T<:AbstractTargetSubject,
    S<:AbstractMCMCState,
}
    algorithm::A
    target::T
    state::S
    info::MCMCChainInfo
end

export MCMCChain



struct MCMCSampleVector{P<:Real,T<:AbstractFloat,W<:Real} <: DenseVector{MCMCSample{P,T,W}}
    params::ExtendableArray{P, 2, 1}
    log_values::Vector{T}
    weights::Vector{W}
end

export MCMCSampleVector

function MCMCSampleVector(chain::MCMCChain)
    P = eltype(chain.state.current_sample.params)
    T = typeof(chain.state.current_sample.log_value)
    W = typeof(chain.state.current_sample.weight)

    m = size(chain.state.current_sample.params, 1)
    MCMCSampleVector(ExtendableArray{P}(m, 0), Vector{T}(0), Vector{W}(0))
end


Base.size(xs::MCMCSampleVector) = size(xs.log_values)

Base.getindex(xs::MCMCSampleVector{P,T,W}, i::Integer) where {P,T,W} =
    MCMCSample{P,T,W}(xs.params[:,i], xs.log_values[i], xs.weights[i])


function Base.push!(xs::MCMCSampleVector, x::MCMCSample)
    append!(xs.params, x.params)
    push!(xs.log_values, x.log_value)
    push!(xs.weights, x.weight)
    xs
end
