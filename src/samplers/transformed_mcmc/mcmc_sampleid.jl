# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type TransformedSampleID end

struct TransformedMCMCTransformedSampleID{
    T<:Int32,
    U<:Int64,
} <: TransformedSampleID
    chainid::T
    chaincycle::T
    stepno::U
end

function TransformedMCMCTransformedSampleID(
    chainid::Integer,
    chaincycle::Integer,
    stepno::Integer,
)
    TransformedMCMCTransformedSampleID(Int32(chainid), Int32(chaincycle), Int64(stepno))
end

const TransformedMCMCTransformedSampleIDVector{TV<:AbstractVector{<:Int32},UV<:AbstractVector{<:Int64}} = StructArray{
    TransformedMCMCTransformedSampleID,
    1,
    NamedTuple{(:chainid, :chaincycle, :stepno), Tuple{TV,TV,UV}},
    Int
}


function TransformedMCMCTransformedSampleIDVector(contents::Tuple{TV,TV,UV}) where {TV<:AbstractVector{<:Int32},UV<:AbstractVector{<:Int64}}
    StructArray{TransformedMCMCTransformedSampleID}(contents)::TransformedMCMCTransformedSampleIDVector{TV,UV}
end

TransformedMCMCTransformedSampleIDVector(::UndefInitializer, len::Integer) = TransformedMCMCTransformedSampleIDVector((
    Vector{Int32}(undef, len), Vector{Int32}(undef, len),
    Vector{Int64}(undef, len)
))

TransformedMCMCTransformedSampleIDVector() = TransformedMCMCTransformedSampleIDVector(undef, 0)


_create_undef_vector(::Type{TransformedMCMCTransformedSampleID}, len::Integer) = TransformedMCMCTransformedSampleIDVector(undef, len)


# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::TransformedMCMCTransformedSampleIDVector, B::TransformedMCMCTransformedSampleIDVector)
    A.chainid == B.chainid &&
    A.chaincycle == B.chaincycle &&
    A.stepno == B.stepno
end


function Base.merge!(X::TransformedMCMCTransformedSampleIDVector, Xs::TransformedMCMCTransformedSampleIDVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::TransformedMCMCTransformedSampleIDVector, Xs::TransformedMCMCTransformedSampleIDVector...) = merge!(deepcopy(X), Xs...)
