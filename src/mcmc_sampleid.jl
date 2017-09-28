# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MCMCSampleID
    chainid::Int32
    chaincycle::Int32
    sampleno::Int64
end

MCMCSampleID(chain::MCMCIterator) =
    MCMCSampleID(chain.id, chain.cycle, current_sampleno(chain))


reset_rng_counters!(rng::AbstractRNG, tags::MCMCSampleID) =
    reset_rng_counters!(rng, tags.chainid, tags.chaincycle, tags.sampleno)


function next_cycle!(chain::MCMCIterator)
    next_cycle!(chain.state)
    chain.cycle += 1
    sampleid = MCMCSampleID(chain)
    @assert sampleid.sampleno == 1
    reset_rng_counters!(chain.rng, sampleid)
    chain
end



struct MCMCSampleIDVector <: BATDataVector{MCMCSampleID}
    chainid::Vector{Int32}
    chaincycle::Vector{Int32}
    sampleno::Vector{Int64}
end

export MCMCSampleIDVector

MCMCSampleIDVector() =
    MCMCSampleIDVector(Vector{Int32}(), Vector{Int32}(), Vector{Int64}())

MCMCSampleIDVector(chain::MCMCIterator) = MCMCSampleIDVector()


Base.size(xs::MCMCSampleIDVector) = size(xs.chainid)

Base.getindex(xs::MCMCSampleIDVector, i::Integer)  =
    MCMCSampleID(xs.chainid[i], xs.chaincycle[i], xs.sampleno[i])

function Base.push!(xs::MCMCSampleIDVector, x::MCMCSampleID)
    push!(xs.chainid, x.chainid)
    push!(xs.chaincycle, x.chaincycle)
    push!(xs.sampleno, x.sampleno)
    xs
end

Base.push!(xs::MCMCSampleIDVector, chain::MCMCIterator) =
    push!(xs, MCMCSampleID(chain))

function Base.append!(A::MCMCSampleIDVector, B::MCMCSampleIDVector)
    append!(A.chainid, B.chainid)
    append!(A.chaincycle, B.chaincycle)
    append!(A.sampleno, B.sampleno)
    A
end
