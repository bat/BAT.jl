# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractDensitySample end
export AbstractDensitySample


struct DensitySample{
    P<:Real,
    T<:Real,
    W<:Real,
    R,
    PA<:AbstractVector{P}
} <: AbstractDensitySample
    params::PA
    log_posterior::T
    log_prior::T
    weight::W
    info::R
end

export DensitySample


# DensitySample behaves as a scalar type under broadcasting:
@inline Base.Broadcast.broadcastable(shape::DensitySample) = Ref(shape)


import Base.==
function ==(A::DensitySample, B::DensitySample)
    A.params == B.params && A.log_posterior == B.log_posterior &&
        A.log_prior == B.log_prior && A.weight == B.weight &&
        A.info == B.info
end


function Base.similar(s::DensitySample{P,T,W,R}) where {P,T,W,R}
    params = fill!(similar(s.params), oob(eltype(s.params)))
    log_posterior = convert(T, NaN)
    log_prior = convert(T, NaN)
    weight = zero(W)
    info = R()
    PA = typeof(params)
    DensitySample{P,T,W,R,PA}(params, log_posterior, log_prior, weight, info)
end


nparams(s::DensitySample) = length(s.params)


function _apply_shape(shape::AbstractValueShape, s::DensitySample)
    (
        params = shape(s.params),
        log_posterior = s.log_posterior,
        log_prior = s.log_prior,
        weight = s.weight,
        info = s.info
    )
end

@static if VERSION >= v"1.3"
    (shape::AbstractValueShape)(s::DensitySample) = _apply_shape(shape, s)
else
    (shape::ScalarShape)(s::DensitySample) = _apply_shape(shape, s)
    (shape::ArrayShape)(s::DensitySample) = _apply_shape(shape, s)
    (shape::ConstValueShape)(s::DensitySample) = _apply_shape(shape, s)
    (shape::NamedTupleShape)(s::DensitySample) = _apply_shape(shape, s)
end



const DensitySampleVector{
    P<:Real,T<:AbstractFloat,W<:Real,R,PA<:AbstractVector{P},
    PAV<:AbstractVector{<:AbstractVector{P}},TV<:AbstractVector{T},WV<:AbstractVector{W},RV<:AbstractVector{R}
} = StructArray{
    DensitySample{P,T,W,R,PA},
    1,
    NamedTuple{(:params, :log_posterior, :log_prior, :weight, :info), Tuple{PAV,TV,TV,WV,RV}}
}

export DensitySampleVector


function StructArray{DensitySample}(
    contents::Tuple{
        AbstractVector{<:AbstractVector{P}},
        AbstractVector{T},
        AbstractVector{T},
        AbstractVector{W},
        AbstractVector{R}
    }
) where {P<:Real,T<:AbstractFloat,W<:Real,R}
    params, log_posterior, log_prior, weight, info = contents
    PA = eltype(params)
    StructArray{DensitySample{P,T,W,R,PA}}(contents)
end


DensitySampleVector(contents::NTuple{5,Any}) = StructArray{DensitySample}(contents)


_create_undef_vector(::Type{T}, len::Integer) where T = Vector{T}(undef, len)


function DensitySampleVector{P,T,W,R}(::UndefInitializer, len::Integer, npar::Integer) where {P<:Real,T<:AbstractFloat,W<:Real,R}
    contents = (
        VectorOfSimilarVectors(ElasticArray{P}(undef, npar, len)),
        Vector{T}(undef, len),
        Vector{T}(undef, len),
        Vector{W}(undef, len),
        _create_undef_vector(R, len)
    )

    DensitySampleVector(contents)
end

DensitySampleVector(::Type{S}, nparams::Integer) where {P<:Real,T<:AbstractFloat,W<:Real,R,S<:DensitySample{P,T,W,R}} =
    DensitySampleVector{P,T,W,R}(undef, 0, nparams)


# Specialize getindex to properly support ArraysOfArrays, preventing
# conversion to exact element type:
@inline Base.getindex(A::StructArray{<:DensitySample}, I::Int...) =
    DensitySample(A.params[I...], A.log_posterior[I...], A.log_prior[I...], A.weight[I...], A.info[I...])

# Specialize IndexStyle, current default for StructArray seems to be IndexCartesian()
Base.IndexStyle(::StructArray{<:DensitySample, 1}) = IndexLinear()

# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::DensitySampleVector, B::DensitySampleVector)
    A.params == B.params &&
    A.log_posterior == B.log_posterior &&
    A.log_prior == B.log_prior &&
    A.weight == B.weight &&
    A.info == B.info
end


function Base.merge!(X::DensitySampleVector, Xs::DensitySampleVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::DensitySampleVector, Xs::DensitySampleVector...) = merge!(deepcopy(X), Xs...)


function UnsafeArrays.uview(A::DensitySampleVector)
    DensitySampleVector((
        uview(A.params),
        uview(A.log_posterior),
        uview(A.log_prior),
        uview(A.weight),
        uview(A.info)
    ))
end


Base.@propagate_inbounds function _bcasted_apply(shape::AbstractValueShape, A::DensitySampleVector)
    TypedTables.Table(
        params = shape.(A.params),
        log_posterior = A.log_posterior,
        log_prior = A.log_prior,
        weight = A.weight,
        info = A.info
    )
end

Base.copy(
    instance::Base.Broadcast.Broadcasted{
        <:Base.Broadcast.AbstractArrayStyle{1},
        <:Any,
        <:AbstractValueShape,
        <:Tuple{DensitySampleVector}
    }
) = _bcasted_apply(instance.f, instance.args[1])    
