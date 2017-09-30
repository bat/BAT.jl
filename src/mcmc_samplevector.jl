# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type MCMCDataVector{T} <: DenseVector{T} end

mcmc_callback(X::MCMCDataVector, args...) = MCMCPushCallback(X, args...)

function Base.merge!(X::MCMCDataVector, Xs::MCMCDataVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::MCMCDataVector, Xs::MCMCDataVector...) = merge!(deepcopy(X), Xs...)



struct MCMCSampleVector{P<:Real,T<:AbstractFloat,W<:Real} <: MCMCDataVector{MCMCSample{P,T,W}}
    params::ExtendableArray{P, 2, 1}
    log_value::Vector{T}
    weight::Vector{W}
end

export MCMCSampleVector

function MCMCSampleVector(chain::MCMCChain)
    P = eltype(chain.state.current_sample.params)
    T = typeof(chain.state.current_sample.log_value)
    W = typeof(chain.state.current_sample.weight)

    m = size(chain.state.current_sample.params, 1)
    MCMCSampleVector(ExtendableArray{P}(m, 0), Vector{T}(0), Vector{W}(0))
end


Base.size(xs::MCMCSampleVector) = size(xs.log_value)

Base.getindex(xs::MCMCSampleVector{P,T,W}, i::Integer) where {P,T,W} =
    MCMCSample{P,T,W}(xs.params[:,i], xs.log_value[i], xs.weight[i])


function Base.push!(xs::MCMCSampleVector, x::MCMCSample)
    append!(xs.params, x.params)
    push!(xs.log_value, x.log_value)
    push!(xs.weight, x.weight)
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

function Base.append!(A::MCMCSampleVector, B::MCMCSampleVector)
    append!(A.params, B.params)
    append!(A.log_value, B.log_value)
    append!(A.weight, B.weight)
    A
end



struct MCMCSampleIDVector <: MCMCDataVector{MCMCSampleID}
    chainid::Vector{Int32}
    chaincycle::Vector{Int32}
    sampleno::Vector{Int64}
end

export MCMCSampleIDVector

MCMCSampleIDVector() =
    MCMCSampleIDVector(Vector{Int32}(), Vector{Int32}(), Vector{Int64}())


Base.size(xs::MCMCSampleIDVector) = size(xs.chainid)

Base.getindex(xs::MCMCSampleIDVector, i::Integer)  =
    MCMCSampleID(xs.chainid[i], xs.chaincycle[i], xs.sampleno[i])

function Base.push!(xs::MCMCSampleIDVector, x::MCMCSampleID)
    push!(xs.chainid, x.chainid)
    push!(xs.chaincycle, x.chaincycle)
    push!(xs.sampleno, x.sampleno)
    xs
end

Base.push!(xs::MCMCSampleIDVector, chain::MCMCChain) =
    push!(xs, MCMCSampleID(chain))

function Base.append!(A::MCMCSampleIDVector, B::MCMCSampleIDVector)
    append!(A.chainid, B.chainid)
    append!(A.chaincycle, B.chaincycle)
    append!(A.sampleno, B.sampleno)
    A
end
