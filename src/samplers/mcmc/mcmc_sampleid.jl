# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const CURRENT_SAMPLE = -1
const PROPOSED_SAMPLE = -2
const INVALID_SAMPLE = 0
const ACCEPTED_SAMPLE = 1
const REJECTED_SAMPLE = 2


struct MCMCSampleID
    chainid::Int32
    chaincycle::Int32
    stepno::Int64
    sampletype::Int64
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


function Base.resize!(A::MCMCSampleIDVector, n::Integer)
    resize!(A.chainid, n)
    resize!(A.chaincycle, n)
    resize!(A.stepno, n)
    resize!(A.sampletype, n)
    A
end


Base.@propagate_inbounds function Base.unsafe_view(A::MCMCSampleIDVector, idxs)
    MCMCSampleIDVector(
        view(A.chainid, idxs),
        view(A.chaincycle, idxs),
        view(A.stepno, idxs),
        view(A.sampletype, idxs)
    )
end


function UnsafeArrays.uview(A::MCMCSampleIDVector)
    MCMCSampleIDVector(
        uview(A.chainid),
        uview(A.chaincycle),
        uview(A.stepno),
        uview(A.sampletype)
    )
end


Tables.istable(::Type{<:MCMCSampleIDVector}) = true

Tables.columnaccess(::Type{<:MCMCSampleIDVector}) = true

Tables.columns(A::MCMCSampleIDVector) = (
    chainid = A.chainid,
    chaincycle = A.chaincycle,
    stepno = A.stepno,
    sampletype = A.sampletype
)

Tables.rowaccess(::Type{<:MCMCSampleIDVector}) = true

Tables.rows(A::MCMCSampleIDVector) = A

Tables.schema(A::MCMCSampleIDVector) = Tables.Schema(
    (:chainid, :chaincycle, :stepno, :sampletype),
    (eltype(A.chainid), eltype(A.chaincycle), eltype(A.stepno), eltype(A.sampletype))
)


function read_fom_hdf5(input, ::Type{MCMCSampleIDVector})
    MCMCSampleIDVector(
        input["chainid"][:],
        input["chaincycle"][:],
        input["stepno"][:],
        input["sampletype"][:],
    )
end


function write_to_hdf5(output, sampleids::MCMCSampleIDVector)
    output["chainid"] = sampleids.chainid
    output["chaincycle"] = sampleids.chaincycle
    output["stepno"] = sampleids.stepno
    output["sampletype"] = sampleids.sampletype
    nothing
end
