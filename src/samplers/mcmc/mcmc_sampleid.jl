# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type SampleID end

struct MCMCSampleID <: SampleID
    chainid::Int32
    walkerid::Int32
    chaincycle::Int32
    stepno::Int64
    proposalid::Int32
    sampletype::Bool
end


const MCMCSampleIDVector{
    TV<:AbstractVector{<:Int32},
    UV<:AbstractVector{<:Int64},
    BV<:AbstractVector{<:Bool}
    } = StructArray{
    MCMCSampleID,
    1,
    NamedTuple{
        (
            :chainid, 
            :walkerid, 
            :chaincycle, 
            :stepno,
            :proposalid,
            :sampletype
        ), 
        Tuple{TV,TV,TV,UV,TV,BV}},
    Int
}


function MCMCSampleIDVector(contents::Tuple{TV,TV,TV,UV,TV,BV}) where {
    TV<:AbstractVector{<:Int32},
    UV<:AbstractVector{<:Int64},
    BV<:AbstractVector{<:Bool}
    }
    StructArray{MCMCSampleID}(contents)::MCMCSampleIDVector{TV,UV,BV}
end

MCMCSampleIDVector(::UndefInitializer, len::Integer) = MCMCSampleIDVector(
    (
        Vector{Int32}(undef, len),
        Vector{Int32}(undef, len),
        Vector{Int32}(undef, len),
        Vector{Int64}(undef, len),
        Vector{Int32}(undef, len),
        Vector{Bool}(undef, len)
    )
)

MCMCSampleIDVector() = MCMCSampleIDVector(undef, 0)


_create_undef_vector(::Type{MCMCSampleID}, len::Integer) = MCMCSampleIDVector(undef, len)


# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::MCMCSampleIDVector, B::MCMCSampleIDVector)
    A.chainid == B.chainid &&
    A.walkerid == B.walkerid &&
    A.chaincycle == B.chaincycle &&
    A.stepno == B.stepno &&
    A.proposalid == B.proposalid &&
    A.sampletype == B.sampletype
end


function Base.merge!(X::MCMCSampleIDVector, Xs::MCMCSampleIDVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::MCMCSampleIDVector, Xs::MCMCSampleIDVector...) = merge!(deepcopy(X), Xs...)
