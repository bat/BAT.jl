# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractDensitySample end
export AbstractDensitySample


# ToDo: Make DensitySample immutable
mutable struct DensitySample{
    P<:Real,
    T<:Real,
    W<:Real,
    PA<:AbstractVector{P}
} <: AbstractDensitySample
    params::PA
    log_value::T
    weight::W
end

export DensitySample


import Base.==
==(A::DensitySample, B::DensitySample) =
    A.params == B.params && A.log_value == B.log_value && A.weight == B.weight


# ToDo: remove?
Base.length(s::DensitySample) = length(s.params)


function Base.similar(s::DensitySample{P,T,W}) where {P,T,W}
    params = fill!(similar(s.params), oob(eltype(s.params)))
    log_value = convert(T, NaN)
    weight = zero(W)
    PA = typeof(params)
    DensitySample{P,T,W,PA}(params, log_value, weight)
end


function Base.copyto!(dest::DensitySample, src::DensitySample)
    copyto!(dest.params, src.params)
    dest.log_value = src.log_value
    dest.weight = src.weight
    dest
end


nparams(s::DensitySample) = length(s)


# ToDo: Make immutable
struct DensitySampleVector{
    P<:Real,T<:AbstractFloat,W<:Real,
    PA<:VectorOfSimilarVectors{P},TA<:AbstractVector{T},WA<:AbstractVector{W}
} <: BATDataVector{DensitySample{P,T,W,Vector{P}}}
    params::PA
    log_value::TA
    weight::WA
end

export DensitySampleVector


function DensitySampleVector{P,T,W}(nparams::Integer) where {P<:Real,T<:AbstractFloat,W<:Real}
    DensitySampleVector(
        VectorOfSimilarVectors(ElasticArray{P}(undef, nparams, 0)),
        Vector{T}(undef, 0),
        Vector{W}(undef, 0)
    )
end

DensitySampleVector(::Type{S}, nparams::Integer) where {P<:Real,T<:AbstractFloat,W<:Real,S<:DensitySample{P,T,W}} =
    DensitySampleVector{P,T,W}(nparams)


Base.size(xs::DensitySampleVector) = size(xs.log_value)

Base.getindex(xs::DensitySampleVector, i::Integer) =
    DensitySample(xs.params[i], xs.log_value[i], xs.weight[i])

Base.@propagate_inbounds Base._getindex(l::IndexStyle, xs::DensitySampleVector, idxs::AbstractVector{<:Integer}) =
    DensitySampleVector(xs.params[:, idxs], xs.log_value[idxs], xs.weight[idxs])

Base.IndexStyle(xs::DensitySampleVector) = IndexStyle(xs.log_value)


function Base.push!(xs::DensitySampleVector, x::DensitySample)
    push!(xs.params, x.params)
    push!(xs.log_value, x.log_value)
    push!(xs.weight, x.weight)
    xs
end


function Base.append!(A::DensitySampleVector, B::DensitySampleVector)
    append!(A.params, B.params)
    append!(A.log_value, B.log_value)
    append!(A.weight, B.weight)
    A
end


Base.@propagate_inbounds function Base.view(A::DensitySampleVector, idxs)
    DensitySampleVector(
        view(A.params, idxs),
        view(A.log_value, idxs),
        view(A.weight, idxs)
    )
end


function _swap!(A::DensitySampleVector, i_A::SingleArrayIndex, B::DensitySampleVector, i_B::SingleArrayIndex)
    _swap!(view(flatview(A.params), :, i_A), view(flatview(A.params), :, i_B))  # Memory allocation!
    _swap!(A.log_value, i_A, B.log_value, i_B)
    _swap!(A.weight, i_A, B.weight, i_B)
    A
end


function read_fom_hdf5(input, ::Type{DensitySampleVector})
    DensitySampleVector(
        VectorOfSimilarVectors(input["params"][:,:]),
        input["log_value"][:],
        input["weight"][:]
    )
end


function write_to_hdf5(output, samples::DensitySampleVector)
    output["params"] = Array(flatview(samples.params))
    output["log_value"] = samples.log_value
    output["weight"] = samples.weight
    nothing
end
