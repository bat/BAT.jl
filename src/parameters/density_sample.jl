# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractDensitySample end
export AbstractDensitySample


struct PosteriorSample{
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

export PosteriorSample


# PosteriorSample behaves as a scalar type under broadcasting:
@inline Base.Broadcast.broadcastable(shape::PosteriorSample) = Ref(shape)


import Base.==
function ==(A::PosteriorSample, B::PosteriorSample)
    A.params == B.params && A.log_posterior == B.log_posterior &&
        A.log_prior == B.log_prior && A.weight == B.weight &&
        A.info == B.info
end


function Base.similar(s::PosteriorSample{P,T,W,R}) where {P,T,W,R}
    params = fill!(similar(s.params), oob(eltype(s.params)))
    log_posterior = convert(T, NaN)
    log_prior = convert(T, NaN)
    weight = zero(W)
    info = R()
    PA = typeof(params)
    PosteriorSample{P,T,W,R,PA}(params, log_posterior, log_prior, weight, info)
end


nparams(s::PosteriorSample) = length(s.params)


function _apply_shape(shape::AbstractValueShape, s::PosteriorSample)
    (
        params = shape(s.params),
        log_posterior = s.log_posterior,
        log_prior = s.log_prior,
        weight = s.weight,
        info = s.info
    )
end

@static if VERSION >= v"1.3"
    (shape::AbstractValueShape)(s::PosteriorSample) = _apply_shape(shape, s)
else
    (shape::ScalarShape)(s::PosteriorSample) = _apply_shape(shape, s)
    (shape::ArrayShape)(s::PosteriorSample) = _apply_shape(shape, s)
    (shape::ConstValueShape)(s::PosteriorSample) = _apply_shape(shape, s)
    (shape::NamedTupleShape)(s::PosteriorSample) = _apply_shape(shape, s)
end



"""
    PosteriorSampleVector

Type alias for `StructArrays.StructArray{<:PosteriorSample,...}`.
"""
const PosteriorSampleVector{
    P<:Real,T<:AbstractFloat,W<:Real,R,PA<:AbstractVector{P},
    PAV<:AbstractVector{<:AbstractVector{P}},TV<:AbstractVector{T},WV<:AbstractVector{W},RV<:AbstractVector{R}
} = StructArray{
    PosteriorSample{P,T,W,R,PA},
    1,
    NamedTuple{(:params, :log_posterior, :log_prior, :weight, :info), Tuple{PAV,TV,TV,WV,RV}}
}

export PosteriorSampleVector


function StructArray{PosteriorSample}(
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
    StructArray{PosteriorSample{P,T,W,R,PA}}(contents)
end


PosteriorSampleVector(contents::NTuple{5,Any}) = StructArray{PosteriorSample}(contents)


_create_undef_vector(::Type{T}, len::Integer) where T = Vector{T}(undef, len)


function PosteriorSampleVector{P,T,W,R}(::UndefInitializer, len::Integer, npar::Integer) where {P<:Real,T<:AbstractFloat,W<:Real,R}
    contents = (
        VectorOfSimilarVectors(ElasticArray{P}(undef, npar, len)),
        Vector{T}(undef, len),
        Vector{T}(undef, len),
        Vector{W}(undef, len),
        _create_undef_vector(R, len)
    )

    PosteriorSampleVector(contents)
end

PosteriorSampleVector(::Type{S}, nparams::Integer) where {P<:Real,T<:AbstractFloat,W<:Real,R,S<:PosteriorSample{P,T,W,R}} =
    PosteriorSampleVector{P,T,W,R}(undef, 0, nparams)


# Specialize getindex to properly support ArraysOfArrays, preventing
# conversion to exact element type:
@inline Base.getindex(A::StructArray{<:PosteriorSample}, I::Int...) =
    PosteriorSample(A.params[I...], A.log_posterior[I...], A.log_prior[I...], A.weight[I...], A.info[I...])

# Specialize IndexStyle, current default for StructArray seems to be IndexCartesian()
Base.IndexStyle(::StructArray{<:PosteriorSample, 1}) = IndexLinear()

# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::PosteriorSampleVector, B::PosteriorSampleVector)
    A.params == B.params &&
    A.log_posterior == B.log_posterior &&
    A.log_prior == B.log_prior &&
    A.weight == B.weight &&
    A.info == B.info
end


function Base.merge!(X::PosteriorSampleVector, Xs::PosteriorSampleVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::PosteriorSampleVector, Xs::PosteriorSampleVector...) = merge!(deepcopy(X), Xs...)


function UnsafeArrays.uview(A::PosteriorSampleVector)
    PosteriorSampleVector((
        uview(A.params),
        uview(A.log_posterior),
        uview(A.log_prior),
        uview(A.weight),
        uview(A.info)
    ))
end


Base.@propagate_inbounds function _bcasted_apply(shape::AbstractValueShape, A::PosteriorSampleVector)
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
        <:Tuple{PosteriorSampleVector}
    }
) = _bcasted_apply(instance.f, instance.args[1])    


"""
    bat_stats(samples::PosteriorSampleVector)

Calculated parameter statistics on `samples`. Returns a
`NamedTuple{(:mode,:mean,:cov,...)}`. Result properties not listed here
are not part of the stable BAT API and subject to change.
"""
function bat_stats(samples::PosteriorSampleVector)
    par_mean = mean(samples.params, FrequencyWeights(samples.weight))
    par_mode_idx = findmax(samples.log_posterior)[2]
    par_mode = samples.params[par_mode_idx]
    par_cov = cov(samples.params, FrequencyWeights(samples.weight))

    (
        mode = par_mode,
        mean = par_mean,
        cov = par_cov
    )
end

export bat_stats


"""
    drop_low_weight_samples(
        samples::PosteriorSampleVector,
        fraction::Real = 10^-4
    )

Drop `fraction` of the total probability mass from samples to filter out the
samples with the lowest weight.

Note: BAT-internal function, not part of stable API.
"""
function drop_low_weight_samples(samples::PosteriorSampleVector, fraction::Real = 10^-5)
    W = float(samples.weight)
    if minimum(W) / maximum(W) > 10^-2
        samples
    else
        W_s = sort(W)
        Q = cumsum(W_s)
        Q ./= maximum(Q)
        @assert last(Q) ≈ 1
        thresh = W_s[searchsortedlast(Q, fraction)]
        idxs = findall(x -> x >= thresh, samples.weight)
        samples[idxs]
    end
end
