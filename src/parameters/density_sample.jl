# This file is a part of BAT.jl, licensed under the MIT License (MIT).


const _default_float_WT = Float64 # Default type for float weights
const _default_int_WT = Int # Default type for int weights
const _default_LDT = Float64 # Default type for log-density values


"""
    struct DensitySample

A weighted sample drawn according to an statistical density,
e.g. a [`BAT.AbstractDensity`](@ref).

Fields:
    * `v`: Multivariate parameter vector
    * `logd`: log of the value of the density at `v`
    * `weight`: Weight of the sample
    * `info`: Additional info on the provenance of the sample. Content depends
       on the sampling algorithm.
    * aux: Custom user-defined information attatched to the sample.

Constructors:

```julia
DensitySample(
    v::AbstractVector{<:Real},
    logd::Real,
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
    v::P
    logd::T
    weight::W
    info::R
    aux::Q
end

export DensitySample


# DensitySample behaves as a scalar type under broadcasting:
@inline Base.Broadcast.broadcastable(shape::DensitySample) = Ref(shape)


import Base.==
function ==(A::DensitySample, B::DensitySample)
    A.v == B.v && A.logd == B.logd &&
        A.weight == B.weight && A.info == B.info && A.aux == B.aux
end


function Base.similar(s::DensitySample{P,T,W,R,Q}) where {P<:AbstractVector{<:Real},T,W,R,Q}
    v = fill!(similar(s.v), oob(eltype(s.v)))
    logd = convert(T, NaN)
    weight = zero(W)
    info = R()
    aux = Q()
    P_new = typeof(v)
    DensitySample{P_new,T,W,R,Q}(v, logd, weight, info, aux)
end


function _apply_shape(shape::AbstractValueShape, s::DensitySample)
    DensitySample(
        stripscalar(shape(s.v)),
        s.logd,
        s.weight,
        s.info,
        s.aux,
    )
end

(shape::AbstractValueShape)(s::DensitySample) = _apply_shape(shape, s)



"""
    DensitySampleVector

Type alias for `StructArrays.StructArray{<:DensitySample,...}`.

Constructor:

```julia
    DensitySampleVector(
        (
            v::AbstractVector{<:AbstractVector{<:Real}}
            logd::AbstractVector{<:Real}
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
    NamedTuple{(:v, :logd, :weight, :info, :aux), Tuple{PV,TV,WV,RV,QV}}
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
    v, logd, weight, info, aux = contents
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

DensitySampleVector(::Type{S}, varlen::Integer) where {P<:AbstractVector{<:Real},T<:AbstractFloat,W<:Real,R,Q,S<:DensitySample{P,T,W,R,Q}} =
    DensitySampleVector{P,T,W,R,Q}(undef, 0, varlen)


# Specialize getindex to properly support ArraysOfArrays and similar, preventing
# conversion to exact element type:

@inline Base.getindex(A::StructArray{<:DensitySample}, I::Int...) =
    DensitySample(A.v[I...], A.logd[I...], A.weight[I...], A.info[I...], A.aux[I...])

@inline Base.getindex(A::StructArray{<:DensitySample}, I::Union{Int,AbstractArray,Colon}...) =
    DensitySampleVector((A.v[I...], A.logd[I...], A.weight[I...], A.info[I...], A.aux[I...]))

# Specialize IndexStyle, current default for StructArray seems to be IndexCartesian()
Base.IndexStyle(::StructArray{<:DensitySample, 1}) = IndexLinear()

# Specialize comparison, currently StructArray seems fall back to `(==)(A::AbstractArray, B::AbstractArray)`
import Base.==
function(==)(A::DensitySampleVector, B::DensitySampleVector)
    A.v == B.v &&
    A.logd == B.logd &&
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
        uview(A.v),
        uview(A.logd),
        uview(A.weight),
        uview(A.info),
        uview(A.aux)
    ))
end


Base.@propagate_inbounds function _bcasted_apply_to_params(f, A::DensitySampleVector)
    DensitySampleVector((
        f.(A.v),
        A.logd,
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


ValueShapes.varshape(A::DensitySampleVector) = elshape(A.v)


function _get_statw(f::Function, samples::DensitySampleVector)
    shape = varshape(samples)
    X = unshaped.(samples.v)
    w = FrequencyWeights(samples.weight)
    r_unshaped = f(X, w)
    shape(r_unshaped)
end

Statistics.mean(samples::DensitySampleVector) = _get_statw(mean, samples)
Statistics.var(samples::DensitySampleVector) = _get_statw(var, samples)

function Statistics.median(samples::DensitySampleVector)
    shape = varshape(samples)
    flat_samples = flatview(unshaped.(samples.v))
    n_params = size(flat_samples)[1]
    median_params = Vector{Float64}()

    for param in Base.OneTo(n_params)
        median_param = quantile(flat_samples[param,:], FrequencyWeights(samples.weight), 0.5)
        push!(median_params, median_param)
    end
    shape(median_params)
end

function Statistics.std(samples::DensitySampleVector)
    shape = varshape(samples)
    X = unshaped.(samples.v)
    w = FrequencyWeights(samples.weight)
    r_unshaped = sqrt.(var(X, w))
    shape(r_unshaped)
end

function _get_stat(f::Function, samples::DensitySampleVector)
    shape = varshape(samples)
    X = unshaped.(samples.v)
    r_unshaped = f(X)
    shape(r_unshaped)
end

Base.minimum(samples::DensitySampleVector) = _get_stat(minimum, samples)
Base.maximum(samples::DensitySampleVector) = _get_stat(maximum, samples)

Statistics.cov(samples::DensitySampleVector{<:AbstractVector{<:Real}}) = cov(samples.v, FrequencyWeights(samples.weight))
Statistics.cor(samples::DensitySampleVector{<:AbstractVector{<:Real}}) = cor(samples.v, FrequencyWeights(samples.weight))

function _get_mode(samples::DensitySampleVector)
    shape = varshape(samples)
    i = findmax(samples.logd)[2]
    v_unshaped = unshaped.(samples.v)[i]
    v = copy(shape(v_unshaped))
    (v, i)
end


StatsBase.mode(samples::DensitySampleVector) = _get_mode(samples)[1]


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
        ind = searchsortedlast(Q, fraction)
        if ind !== 0
            thresh = W_s[searchsortedlast(Q, fraction)]
            idxs = findall(x -> x >= thresh, samples.weight)
            samples[idxs]
        else
            samples
        end
    end
end
