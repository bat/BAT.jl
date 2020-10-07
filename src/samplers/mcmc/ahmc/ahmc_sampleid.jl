# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct AHMCSampleID <: SampleID
    chainid::Int32
    chaincycle::Int32
    stepno::Int64
    sampletype::Int32
    hamiltonian_energy::Float64
    tree_depth::Int32
    divergent::Bool
    step_size::Float64     
end


const AHMCSampleIDVector{TV<:AbstractVector{<:Int32},UV<:AbstractVector{<:Int64},
                         FV<:AbstractVector{<:Float64},BV<:AbstractVector{<:Bool}} = StructArray{
    AHMCSampleID,
    1,
    NamedTuple{
        (:chainid, :chaincycle, :stepno, :sampletype, :hamiltonian_energy, :tree_depth, :divergent, :step_size),
        Tuple{TV,TV,UV,TV,FV,TV,BV,FV}
    },
    Int
}


AHMCSampleIDVector(contents::NTuple{8,Any}) = StructArray{AHMCSampleID}(contents)

AHMCSampleIDVector(::UndefInitializer, len::Integer) = AHMCSampleIDVector((
    Vector{Int32}(undef, len), Vector{Int32}(undef, len),
    Vector{Int64}(undef, len), Vector{Int64}(undef, len),
    Vector{Float64}(undef, len), Vector{Int64}(undef, len),
    Vector{Bool}(undef, len), Vector{Float64}(undef, len),
))

AHMCSampleIDVector() = AHMCSampleIDVector(undef, 0)


_create_undef_vector(::Type{AHMCSampleID}, len::Integer) = AHMCSampleIDVector(undef, len)


# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::AHMCSampleIDVector, B::AHMCSampleIDVector)
    A.chainid == B.chainid &&
    A.chaincycle == B.chaincycle &&
    A.stepno == B.stepno &&
    A.sampletype == B.sampletype &&
    A.hamiltonian_energy == B.hamiltonian_energy &&
    A.tree_depth == B.tree_depth &&
    A.divergent == B.divergent &&
    A.step_size == B.stepsize
end


function Base.merge!(X::AHMCSampleIDVector, Xs::AHMCSampleIDVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::AHMCSampleIDVector, Xs::AHMCSampleIDVector...) = merge!(deepcopy(X), Xs...)

