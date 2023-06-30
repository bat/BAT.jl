# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type TransformedSampleID end

struct TransformedMCMCSampleID{
    T<:Int32,
    U<:Int64,
} <: TransformedSampleID
    chainid::T
    chaincycle::T
    stepno::U
end

function TransformedMCMCSampleID(
    chainid::Integer,
    chaincycle::Integer,
    stepno::Integer,
)
    TransformedMCMCSampleID(Int32(chainid), Int32(chaincycle), Int64(stepno))
end

const TransformedMCMCSampleIDVector{TV<:AbstractVector{<:Int32},UV<:AbstractVector{<:Int64}} = StructArray{
    TransformedMCMCSampleID,
    1,
    NamedTuple{(:chainid, :chaincycle, :stepno), Tuple{TV,TV,UV}},
    Int
}


function TransformedMCMCSampleIDVector(contents::Tuple{TV,TV,UV}) where {TV<:AbstractVector{<:Int32},UV<:AbstractVector{<:Int64}}
    StructArray{TransformedMCMCSampleID}(contents)::TransformedMCMCSampleIDVector{TV,UV}
end

TransformedMCMCSampleIDVector(::UndefInitializer, len::Integer) = TransformedMCMCSampleIDVector((
    Vector{Int32}(undef, len), Vector{Int32}(undef, len),
    Vector{Int64}(undef, len)
))

TransformedMCMCSampleIDVector() = TransformedMCMCSampleIDVector(undef, 0)


_create_undef_vector(::Type{TransformedMCMCSampleID}, len::Integer) = TransformedMCMCSampleIDVector(undef, len)


# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::TransformedMCMCSampleIDVector, B::TransformedMCMCSampleIDVector)
    A.chainid == B.chainid &&
    A.chaincycle == B.chaincycle &&
    A.stepno == B.stepno
end


function Base.merge!(X::TransformedMCMCSampleIDVector, Xs::TransformedMCMCSampleIDVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::TransformedMCMCSampleIDVector, Xs::TransformedMCMCSampleIDVector...) = merge!(deepcopy(X), Xs...)
