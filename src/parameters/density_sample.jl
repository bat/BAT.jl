# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const _default_float_WT = Float64 # Default type for float weights
const _default_int_WT = Int # Default type for int weights
const _default_LDT = Float64 # Default type for log-density values


"""
    struct DensitySample

A weighted sample drawn according to an statistical density,
e.g. a [`BAT.AbstractDensity`](@ref).

Fields:
    * `params`: Multivariate parameter vector
    * `logdensity`: log of the value of the density at `params`
    * `weight`: Weight of the sample
    * `info`: Additional info on the provenance of the sample. Content depends
       on the sampling algorithm.
    * aux: Custom user-defined information attatched to the sample.

Constructors:

```julia
DensitySample(
    params::AbstractVector{<:Real},
    logdensity::Real,
    weight::Real,
    info::Any,
    aux::Any
)
```

Use [`DensitySampleVector`](@ref) to store vectors of multiple samples with
an efficient column-based memory layout.
"""
struct DensitySample{
    P,
    T<:Real,
    W<:Real,
    R,
    Q
}
    params::P
    logdensity::T
    weight::W
    info::R
    aux::Q
end

export DensitySample


# DensitySample behaves as a scalar type under broadcasting:
@inline Base.Broadcast.broadcastable(shape::DensitySample) = Ref(shape)


import Base.==
function ==(A::DensitySample, B::DensitySample)
    A.params == B.params && A.logdensity == B.logdensity &&
        A.weight == B.weight && A.info == B.info && A.aux == B.aux
end


function Base.similar(s::DensitySample{P,T,W,R,Q}) where {P<:AbstractVector{<:Real},T,W,R,Q}
    params = fill!(similar(s.params), oob(eltype(s.params)))
    logdensity = convert(T, NaN)
    weight = zero(W)
    info = R()
    aux = Q()
    P_new = typeof(params)
    DensitySample{P_new,T,W,R,Q}(params, logdensity, weight, info, aux)
end


nparams(s::DensitySample) = length(s.params)


function _apply_shape(shape::AbstractValueShape, s::DensitySample)
    DensitySample(
        stripscalar(shape(s.params)),
        s.logdensity,
        s.weight,
        s.info,
        s.aux,
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



"""
    DensitySampleVector

Type alias for `StructArrays.StructArray{<:DensitySample,...}`.

Constructor:

```julia
    DensitySampleVector(
        (
            params::AbstractVector{<:AbstractVector{<:Real}}
            logdensity::AbstractVector{<:Real}
            weight::AbstractVector{<:Real}
            info::AbstractVector{<:Any}
            aux::AbstractVector{<:Any}
        )
    )
```
"""
const DensitySampleVector{
    P,T<:AbstractFloat,W<:Real,R,Q,
    PV<:AbstractVector{P},TV<:AbstractVector{T},WV<:AbstractVector{W},RV<:AbstractVector{R},QV<:AbstractVector{Q}
} = StructArray{
    DensitySample{P,T,W,R,Q},
    1,
    NamedTuple{(:params, :logdensity, :weight, :info, :aux), Tuple{PV,TV,WV,RV,QV}}
}

export DensitySampleVector


function StructArray{DensitySample}(
    contents::Tuple{
        AbstractVector{P},
        AbstractVector{T},
        AbstractVector{W},
        AbstractVector{R},
        AbstractVector{Q},
    }
) where {P,T<:AbstractFloat,W<:Real,R,Q}
    params, logdensity, weight, info, aux = contents
    StructArray{DensitySample{P,T,W,R,Q}}(contents)
end


DensitySampleVector(contents::NTuple{5,Any}) = StructArray{DensitySample}(contents)


_create_undef_vector(::Type{T}, len::Integer) where T = Vector{T}(undef, len)


function DensitySampleVector{P,T,W,R,Q}(::UndefInitializer, len::Integer, npar::Integer) where {
    PT<:Real, P<:AbstractVector{PT}, T<:AbstractFloat, W<:Real, R, Q
}
    contents = (
        VectorOfSimilarVectors(ElasticArray{PT}(undef, npar, len)),
        Vector{T}(undef, len),
        Vector{W}(undef, len),
        _create_undef_vector(R, len),
        _create_undef_vector(Q, len)
    )

    DensitySampleVector(contents)
end

DensitySampleVector(::Type{S}, nparams::Integer) where {P<:AbstractVector{<:Real},T<:AbstractFloat,W<:Real,R,Q,S<:DensitySample{P,T,W,R,Q}} =
    DensitySampleVector{P,T,W,R,Q}(undef, 0, nparams)


# Specialize getindex to properly support ArraysOfArrays, preventing
# conversion to exact element type:
@inline Base.getindex(A::StructArray{<:DensitySample}, I::Int...) =
    DensitySample(A.params[I...], A.logdensity[I...], A.weight[I...], A.info[I...], A.aux[I...])

# Specialize IndexStyle, current default for StructArray seems to be IndexCartesian()
Base.IndexStyle(::StructArray{<:DensitySample, 1}) = IndexLinear()

# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::DensitySampleVector, B::DensitySampleVector)
    A.params == B.params &&
    A.logdensity == B.logdensity &&
    A.weight == B.weight &&
    A.info == B.info &&
    A.aux == B.aux
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
        uview(A.logdensity),
        uview(A.weight),
        uview(A.info),
        uview(A.aux)
    ))
end


Base.@propagate_inbounds function _bcasted_apply_to_params(f, A::DensitySampleVector)
    DensitySampleVector((
        f.(A.params),
        A.logdensity,
        A.weight,
        A.info,
        A.aux
    ))
end

Base.copy(
    instance::Base.Broadcast.Broadcasted{
        <:Base.Broadcast.AbstractArrayStyle{1},
        <:Any,
        <:Union{AbstractValueShape,typeof(unshaped)},
        <:Tuple{DensitySampleVector}
    }
) = _bcasted_apply_to_params(instance.f, instance.args[1])


ValueShapes.varshape(A::DensitySampleVector) = elshape(A.params)


"""
    bat_stats(samples::DensitySampleVector)

Calculated parameter statistics on `samples`. Returns a
`NamedTuple{(:mode,:mean,:cov,...)}`. Result properties not listed here
are not part of the stable BAT API and subject to change.
"""
function bat_stats(samples::DensitySampleVector)
    par_mean = mean(samples.params, FrequencyWeights(samples.weight))
    par_mode_idx = findmax(samples.logdensity)[2]
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
        samples::DensitySampleVector,
        fraction::Real = 10^-4
    )

*BAT-internal, not part of stable public API.*

Drop `fraction` of the total probability mass from samples to filter out the
samples with the lowest weight.
"""
function drop_low_weight_samples(samples::DensitySampleVector, fraction::Real = 10^-5)
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
