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


function Base.push!(xs::MCMCSampleVector, state::AbstractMCMCState)
    if sample_available(state, Val(:any))
        push!(xs, current_sample(state, Val(:any)))
    end
    xs
end


function Base.push!(xs::MCMCSampleVector, chain::MCMCChain)
    push!(xs, chain.state)
    chain
end


# ToDo: merge/append for MCMCSampleVector




struct MCMCSampleVectorCallback{SV<:MCMCSampleVector} <: AbstractMCMCCallback
    samples::SV
    max_level::Int
end

MCMCSampleVectorCallback(sv::MCMCSampleVector) = MCMCSampleVectorCallback(sv, 1)

mcmc_callback(sv::MCMCSampleVector, args...) = MCMCSampleVectorCallback(sv, args...)


function (cb::MCMCSampleVectorCallback)(level::Integer, chain::MCMCChain)
    if (level <= cb.max_level)
        push!(cb.samples, chain)
    end
    nothing
end
