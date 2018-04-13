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

DensitySample(params::PA, log_value::T, weight::W) where {
    P<:Real,
    T<:Real,
    W<:Real,
    PA<:AbstractVector{P}
} = DensitySample{P,T,W,PA}(params, log_value, weight)


import Base.==
==(A::DensitySample, B::DensitySample) =
    A.params == B.params && A.log_value == B.log_value && A.weight == B.weight


# ToDo: remove?
Base.length(s::DensitySample) = length(s.params)


function Base.similar(s::DensitySample{P,T,W}) where {P,T,W}
    params = oob(s.params)
    log_value = convert(T, NaN)
    weight = zero(W)
    PA = typeof(params)
    DensitySample{P,T,W,PA}(params, log_value, weight)
end


function Base.copy!(dest::DensitySample, src::DensitySample) 
    copy!(dest.params, src.params)
    dest.log_value = src.log_value
    dest.weight = src.weight
    dest
end


nparams(s::DensitySample) = length(s)


# ToDo: Make immutable
struct DensitySampleVector{
    P<:Real,T<:AbstractFloat,W<:Real,
    PA<:AbstractArray{P,2},TA<:AbstractArray{T,1},WA<:AbstractArray{W,1}
} <: BATDataVector{DensitySample{P,T,W,Vector{P}}}
    params::PA
    log_value::TA
    weight::WA
end

export DensitySampleVector


DensitySampleVector(params::PA, log_value::TA, weight::WA) where {
    P<:Real,T<:AbstractFloat,W<:Real,
    PA<:AbstractArray{P,2},TA<:AbstractArray{T,1},WA<:AbstractArray{W,1}
} = DensitySampleVector{P,T,W,PA,TA,WA}(params, log_value, weight)

DensitySampleVector{P,T,W}(nparams::Integer) where {P<:Real,T<:AbstractFloat,W<:Real} =
    DensitySampleVector(ElasticArray{P}(nparams, 0), Vector{T}(0), Vector{W}(0))

DensitySampleVector(::Type{S}, nparams::Integer) where {P<:Real,T<:AbstractFloat,W<:Real,S<:DensitySample{P,T,W}} =
    DensitySampleVector{P,T,W}(nparams)


Base.size(xs::DensitySampleVector) = size(xs.log_value)

Base.getindex(xs::DensitySampleVector, i::Integer) =
    DensitySample(view(xs.params, :, i), xs.log_value[i], xs.weight[i])

Base.@propagate_inbounds Base._getindex(l::IndexStyle, xs::DensitySampleVector, idxs::AbstractVector{<:Integer}) =
    DensitySampleVector(xs.params[:, idxs], xs.log_value[idxs], xs.weight[idxs])

Base.IndexStyle(xs::DensitySampleVector) = IndexStyle(xs.log_value)


function Base.push!(xs::DensitySampleVector, x::DensitySample)
    append!(xs.params, x.params)
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
        view(A.params, :, idxs),
        view(A.log_value, idxs),
        view(A.weight, idxs)
    )
end


function _swap!(A::DensitySampleVector, i_A::SingleArrayIndex, B::DensitySampleVector, i_B::SingleArrayIndex)
    _swap!(view(A.params, :, i_A), view(A.params, :, i_B))  # Memory allocation!
    _swap!(A.log_value, i_A, B.log_value, i_B)
    _swap!(A.weight, i_A, B.weight, i_B)
    A
end
