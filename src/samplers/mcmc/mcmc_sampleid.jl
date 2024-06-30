# This file is a part of BAT.jl, licensed under the MIT License (MIT).

const CURRENT_SAMPLE = -1
const PROPOSED_SAMPLE = -2
const INVALID_SAMPLE = 0
const ACCEPTED_SAMPLE = 1
const REJECTED_SAMPLE = 2

abstract type SampleID end

struct MCMCSampleID{
    T<:Int32,
    U<:Union{Int64, Nothing},
} <: SampleID
    chainid::T
    chaincycle::T
    stepno::U
    sampletype::U
end

function MCMCSampleID(
    chainid::Integer,
    chaincycle::Integer,
    stepno::Integer,
)
    MCMCSampleID(Int32(chainid), Int32(chaincycle), Int64(stepno), nothing)
end

const MCMCSampleIDVector{TV<:AbstractVector{<:Int32},UV<:AbstractVector{<:Union{Int64, Nothing}}} = StructArray{
    MCMCSampleID,
    1,
    NamedTuple{(:chainid, :chaincycle, :stepno, :sampletype), Tuple{TV,TV, UV,UV}},
    Int
}


function MCMCSampleIDVector(contents::Tuple{TV,TV, UV, UV}) where {TV<:AbstractVector{<:Int32},UV<:AbstractVector{<:Union{Int64, Nothing}}}
    StructArray{MCMCSampleID}(contents)::MCMCSampleIDVector{TV,UV}
end

MCMCSampleIDVector(::UndefInitializer, len::Integer) = MCMCSampleIDVector((
    Vector{Int32}(undef, len), Vector{Int32}(undef, len),
    Vector{Union{Int64, Nothing}}(undef, len), Vector{Union{Int64, Nothing}}(undef, len)
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
