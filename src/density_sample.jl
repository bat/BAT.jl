# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractDensitySample end
export AbstractDensitySample



mutable struct DensitySample{
    P<:Real,
    T<:Real,
    W<:Real
} <: AbstractDensitySample
    params::Vector{P}
    log_value::T
    weight::W
end

export DensitySample


Base.length(s::DensitySample) = length(s.params)

Base.similar(s::DensitySample{P,T,W}) where {P,T,W} =
    DensitySample{P,T,W}(oob(s.params), convert(T, NaN), zero(W))

import Base.==
==(A::DensitySample, B::DensitySample) =
    A.params == B.params && A.log_value == B.log_value && A.weight == B.weight


function Base.copy!(dest::DensitySample, src::DensitySample) 
    copy!(dest.params, src.params)
    dest.log_value = src.log_value
    dest.weight = src.weight
    dest
end


nparams(s::DensitySample) = length(s)



struct DensitySampleVector{P<:Real,T<:AbstractFloat,W<:Real} <: BATDataVector{DensitySample{P,T,W}}
    params::ElasticArray{P, 2, 1}
    log_value::Vector{T}
    weight::Vector{W}
end

export DensitySampleVector


Base.size(xs::DensitySampleVector) = size(xs.log_value)

Base.getindex(xs::DensitySampleVector{P,T,W}, i::Integer) where {P,T,W} =
    DensitySample{P,T,W}(xs.params[:,i], xs.log_value[i], xs.weight[i])


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
