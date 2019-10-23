# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractDensitySample end
export AbstractDensitySample


struct DensitySample{
    P<:Real,
    T<:Real,
    W<:Real,
    PA<:AbstractVector{P}
} <: AbstractDensitySample
    params::PA
    log_posterior::T
    log_prior::T
    weight::W
end

export DensitySample


# DensitySample behaves as a scalar type under broadcasting:
@inline Base.Broadcast.broadcastable(shape::DensitySample) = Ref(shape)


import Base.==
function ==(A::DensitySample, B::DensitySample)
    A.params == B.params && A.log_posterior == B.log_posterior &&
        A.log_prior == B.log_prior && A.weight == B.weight
end


function Base.similar(s::DensitySample{P,T,W}) where {P,T,W}
    params = fill!(similar(s.params), oob(eltype(s.params)))
    log_posterior = convert(T, NaN)
    log_prior = convert(T, NaN)
    weight = zero(W)
    PA = typeof(params)
    DensitySample{P,T,W,PA}(params, log_posterior, log_prior, weight)
end


nparams(s::DensitySample) = length(s.params)


function _apply_shape(shape::AbstractValueShape, s::DensitySample)
    (
        params = shape(s.params),
        log_posterior = s.log_posterior,
        log_prior = s.log_prior,
        weight = s.weight,
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
    P<:Real,T<:AbstractFloat,W<:Real,PA<:AbstractVector{P},
    PAV<:AbstractVector{<:AbstractVector{P}},TV<:AbstractVector{T},WV<:AbstractVector{W},
} = StructArray{
    DensitySample{P,T,W,PA},
    1,
    NamedTuple{(:params, :log_posterior, :log_prior, :weight), Tuple{PAV,TV,TV,WV}}
}

export DensitySampleVector


function StructArray{DensitySample}(
    contents::Tuple{
        AbstractVector{<:AbstractVector{P}},
        AbstractVector{T},
        AbstractVector{T},
        AbstractVector{W}
    }
) where {P<:Real,T<:AbstractFloat,W<:Real}
    params, log_posterior, log_prior, weight = contents
    PA = eltype(params)
    StructArray{DensitySample{P,T,W,PA}}(contents)
end


DensitySampleVector(contents::NTuple{4,Any}) = StructArray{DensitySample}(contents)


function DensitySampleVector{P,T,W}(nparams::Integer) where {P<:Real,T<:AbstractFloat,W<:Real}
    DensitySampleVector((
        VectorOfSimilarVectors(ElasticArray{P}(undef, nparams, 0)),
        Vector{T}(undef, 0),
        Vector{T}(undef, 0),
        Vector{W}(undef, 0)
    ))
end

DensitySampleVector(::Type{S}, nparams::Integer) where {P<:Real,T<:AbstractFloat,W<:Real,S<:DensitySample{P,T,W}} =
    DensitySampleVector{P,T,W}(nparams)


# Specialize getindex to properly support ArraysOfArrays, preventing
# conversion to exact element type:
@inline Base.getindex(A::StructArray{<:DensitySample}, I::Int...) =
    DensitySample(A.params[I...], A.log_posterior[I...], A.log_prior[I...], A.weight[I...])

# Specialize IndexStyle, current default for StructArray seems to be IndexCartesian()
Base.IndexStyle(::StructArray{<:DensitySample, 1}) = IndexLinear()

# Specialize IndexStyle, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::DensitySampleVector, B::DensitySampleVector)
    A.params == B.params &&
    A.log_posterior == B.log_posterior &&
    A.log_prior == B.log_prior &&
    A.weight == B.weight
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
        uview(A.weight)
    ))
end


Base.@propagate_inbounds function _bcasted_apply(shape::AbstractValueShape, A::DensitySampleVector)
    TypedTables.Table(
        params = shape.(A.params),
        log_posterior = A.log_posterior,
        log_prior = A.log_prior,
        weight = A.weight
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


function read_fom_hdf5(input, ::Type{DensitySampleVector})
    DensitySampleVector((
        VectorOfSimilarVectors(input["params"][:,:]),
        input["log_posterior"][:],
        input["log_prior"][:],
        input["weight"][:]
    ))
end


function write_to_hdf5(output, samples::DensitySampleVector)
    output["params"] = Array(flatview(samples.params))
    output["log_posterior"] = samples.log_posterior
    output["log_prior"] = samples.log_prior
    output["weight"] = samples.weight
    nothing
end
