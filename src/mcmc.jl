# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCState end

abstract type MCMCAlgorithm{S<:AbstractMCMCState} end


abstract type AbstractMCMCSample end
export AbstractMCMCSample



const Parameters = Val(:Parameters)
const LogValues = Val(:LogValues)
const Weights = Val(:Weights)

export Parameters, LogValues, Weights



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



struct MCMCSampleVector{P<:Real,T<:AbstractFloat,W<:Real}
    m::Int
    params::Vector{P}
    log_values::Vector{T}
    weights::Vector{W}
end

export MCMCSampleVector

function MCMCSampleVector(chain::MCMCChain) #<: DenseVector{MCMCSample}
    P = eltype(chain.state.current_sample.params)
    T = typeof(chain.state.current_sample.log_value)
    W = typeof(chain.state.current_sample.weight)

    m = size(chain.state.current_sample.params, 1)
    MCMCSampleVector(m, P[], T[], W[])
end


#Base.eltype(xs::MCMCSampleVector{P,T,W}) where {P,T,W} = MCMCSample{P,T,W}

Base.size(xs::MCMCSampleVector) = size(xs.log_values)

#function Base.getindex(xs::MCMCSampleVector, i::Integer)
#    result = getindex(input.tchain, i)
#    copy_from_proxies!(input.bindings)
#    result
#end

Base.getindex(xs::MCMCSampleVector, ::typeof(Parameters)) = reshape(view(xs.params, :), xs.m, size(xs.log_values, 1))
Base.getindex(xs::MCMCSampleVector, ::typeof(LogValues)) = xs.log_values
Base.getindex(xs::MCMCSampleVector, ::typeof(Weights)) = xs.weights

function Base.push!(xs::MCMCSampleVector, x::MCMCSample)
    append!(xs.params, x.params)
    push!(xs.log_values, x.log_value)
    push!(xs.weights, x.weight)
    xs
end
