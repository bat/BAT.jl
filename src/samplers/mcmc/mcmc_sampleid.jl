# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const CURRENT_SAMPLE = -1
const PROPOSED_SAMPLE = -2
const INVALID_SAMPLE = 0
const ACCEPTED_SAMPLE = 1
const REJECTED_SAMPLE = 2

abstract type SampleID end

struct MCMCSampleID <: SampleID
    chainid::Int32
    chaincycle::Int32
    stepno::Int64
    sampletype::Int32
end


const MCMCSampleIDVector{TV<:AbstractVector{<:Int32},UV<:AbstractVector{<:Int64}} = StructArray{
    MCMCSampleID,
    1,
    NamedTuple{(:chainid, :chaincycle, :stepno, :sampletype), Tuple{TV,TV,UV,UV}},
    Int
}


MCMCSampleIDVector(contents::NTuple{4,Any}) = StructArray{MCMCSampleID}(contents)

MCMCSampleIDVector(::UndefInitializer, len::Integer) = MCMCSampleIDVector((
    Vector{Int32}(undef, len), Vector{Int32}(undef, len),
    Vector{Int64}(undef, len), Vector{Int64}(undef, len)
))

MCMCSampleIDVector() = MCMCSampleIDVector(undef, 0)


_create_undef_vector(::Type{MCMCSampleID}, len::Integer) = MCMCSampleIDVector(undef, len)


# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::MCMCSampleIDVector, B::MCMCSampleIDVector)
    A.chainid == B.chainid &&
    A.chaincycle == B.chaincycle &&
    A.stepno == B.stepno &&
    A.sampletype == B.sampletype
end


function Base.merge!(X::MCMCSampleIDVector, Xs::MCMCSampleIDVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::MCMCSampleIDVector, Xs::MCMCSampleIDVector...) = merge!(deepcopy(X), Xs...)
