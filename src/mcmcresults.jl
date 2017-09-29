# This file is a part of BAT.jl, licensed under the MIT License (MIT).


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


#=

struct MCMCSampleVectorPusher{SV<:MCMCSampleVector,ST<:Val=Val{:complete}}
    samples::SV
end

MCMCSampleVectorPusher

pusher::MCMCSampleVectorPusher()


=#


#=
abstract type AbstractMCMCResult end

mutable struct MCMCResultSamples{SV<:MCMCSampleVector} <: AbstractMCMCResult
    
end



mutable struct MCMCDiagResults{
    SV<:MCMCSampleVector,
    ST<:AbstractMCMCStats
}
    samples_acc::SV
    samples_rej::SV
    stats::ST
end

MCMCDiagResults(chain::MCMCChain)
    samples_acc = MCMCSampleVector(first(chains))
    samples_rej = MCMCSampleVector(first(chains))
    stats = MCMCBasicStats.(chains)
    MCMCDiagResults(samples, samples_rej, stats)
end


# merge!,merge for AbstractVector{<:MCMCResult}

=#

