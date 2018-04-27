# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct MCMCSampleID
    chainid::Int32
    chaincycle::Int32
    stepno::Int64
    sampletype::Int
end



struct MCMCSampleIDVector{
    VInt32<:AbstractVector{Int32},
    VInt64<:AbstractVector{Int64}
} <: BATDataVector{MCMCSampleID}
    chainid::VInt32
    chaincycle::VInt32
    stepno::VInt64
    sampletype::VInt64
end

export MCMCSampleIDVector

MCMCSampleIDVector() = MCMCSampleIDVector(Vector{Int32}(), Vector{Int32}(), Vector{Int64}(), Vector{Int64}())

MCMCSampleIDVector(chain::MCMCIterator) = MCMCSampleIDVector()


Base.size(xs::MCMCSampleIDVector) = size(xs.chainid)

Base.@propagate_inbounds Base.getindex(xs::MCMCSampleIDVector, i::Integer)  =
    MCMCSampleID(xs.chainid[i], xs.chaincycle[i], xs.stepno[i], xs.sampletype[i])

Base.@propagate_inbounds function Base.setindex!(xs::MCMCSampleIDVector, x::MCMCSampleID, i::Integer)
    xs.chainid[i] = x.chainid
    xs.chaincycle[i] = x.chaincycle
    xs.stepno[i] = x.stepno
    xs.sampletype[i] = x.sampletype
    xs
end

Base.IndexStyle(xs::MCMCSampleIDVector) = IndexStyle(xs.chainid)


function Base.push!(xs::MCMCSampleIDVector, x::MCMCSampleID)
    push!(xs.chainid, x.chainid)
    push!(xs.chaincycle, x.chaincycle)
    push!(xs.stepno, x.stepno)
    push!(xs.sampletype, x.sampletype)
    xs
end


function Base.append!(A::MCMCSampleIDVector, B::MCMCSampleIDVector)
    append!(A.chainid, B.chainid)
    append!(A.chaincycle, B.chaincycle)
    append!(A.stepno, B.stepno)
    append!(A.sampletype, B.sampletype)
    A
end


function Base.append!(A::MCMCSampleIDVector, B::AbstractVector{<:MCMCSampleID})
    for x in B
        push!(A, x)
    end
    A
end


Base.@propagate_inbounds function Base.view(A::MCMCSampleIDVector, idxs)
    MCMCSampleIDVector(
        view(A.chainid, idxs),
        view(A.chaincycle, idxs),
        view(A.stepno, idxs),
        view(A.sampletype, idxs)
    )
end


Base.convert(::Type{AbstractMCMCCallback}, x::MCMCSampleIDVector) = MCMCAppendCallback(x)

MCMCAppendCallback(x::MCMCSampleIDVector, nonzero_weights::Bool = true) =
    MCMCAppendCallback(x, 1, get_sample_ids!, nonzero_weights)
