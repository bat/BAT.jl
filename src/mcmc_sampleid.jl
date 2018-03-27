# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MCMCSampleID
    chainid::Int32
    chaincycle::Int32
    sampletype::Int
    stepno::Int64
end

MCMCSampleID(chain::MCMCIterator, sampletype::Int = 1) =
    MCMCSampleID(chain.id, chain.cycle, sampletype, current_stepno(chain))


reset_rng_counters!(rng::AbstractRNG, tags::MCMCSampleID) =
    reset_rng_counters!(rng, tags.chainid, tags.chaincycle, tags.stepno)


function next_cycle!(chain::MCMCIterator)
    next_cycle!(chain.state)
    chain.cycle += 1
    sampleid = MCMCSampleID(chain)
    @assert sampleid.stepno == 1
    reset_rng_counters!(chain.rng, sampleid)
    chain
end



struct MCMCSampleIDVector <: BATDataVector{MCMCSampleID}
    chainid::Vector{Int32}
    chaincycle::Vector{Int32}
    sampletype::Vector{Int}
    stepno::Vector{Int64}
end

export MCMCSampleIDVector

MCMCSampleIDVector() =
    MCMCSampleIDVector(Vector{Int32}(), Vector{Int32}(), Vector{Int64}())

MCMCSampleIDVector(chain::MCMCIterator) = MCMCSampleIDVector()


Base.size(xs::MCMCSampleIDVector) = size(xs.chainid)

Base.getindex(xs::MCMCSampleIDVector, i::Integer)  =
    MCMCSampleID(xs.chainid[i], xs.chaincycle[i], xs.sampletype[i], xs.stepno[i])

Base.IndexStyle(xs::MCMCSampleIDVector) = IndexStyle(xs.chainid)


function Base.push!(xs::MCMCSampleIDVector, x::MCMCSampleID)
    push!(xs.chainid, x.chainid)
    push!(xs.chaincycle, x.chaincycle)
    push!(xs.sampletype, x.sampletype)
    push!(xs.stepno, x.stepno)
    xs
end

function Base.append!(xs::MCMCSampleIDVector, chain::MCMCIterator)
    for i in 1:nsamples_available(chain)
        push!(xs, MCMCSampleID(chain))
    end
end

function Base.append!(A::MCMCSampleIDVector, B::MCMCSampleIDVector)
    append!(A.chainid, B.chainid)
    append!(A.chaincycle, B.chaincycle)
    append!(A.sampletype, B.sampletype)
    append!(A.stepno, B.stepno)
    A
end

Base.convert(::Type{AbstractMCMCCallback}, x::MCMCSampleIDVector) = MCMCAppendCallback(x)
