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


import Base.==
function ==(A::DensitySample, B::DensitySample)
    A.params == B.params && A.log_posterior == B.log_posterior &&
        A.log_prior == B.log_prior && A.weight == B.weight
end


# ToDo: remove?
Base.length(s::DensitySample) = length(s.params)


function Base.similar(s::DensitySample{P,T,W}) where {P,T,W}
    params = fill!(similar(s.params), oob(eltype(s.params)))
    log_posterior = convert(T, NaN)
    log_prior = convert(T, NaN)
    weight = zero(W)
    PA = typeof(params)
    DensitySample{P,T,W,PA}(params, log_posterior, log_prior, weight)
end


nparams(s::DensitySample) = length(s)


struct DensitySampleVector{
    P<:Real,T<:AbstractFloat,W<:Real,
    PA<:VectorOfSimilarVectors{P},TA<:AbstractVector{T},WA<:AbstractVector{W}
} <: BATDataVector{DensitySample{P,T,W,Vector{P}}}
    params::PA
    log_posterior::TA
    log_prior::TA
    weight::WA
end

export DensitySampleVector


function DensitySampleVector{P,T,W}(nparams::Integer) where {P<:Real,T<:AbstractFloat,W<:Real}
    DensitySampleVector(
        VectorOfSimilarVectors(ElasticArray{P}(undef, nparams, 0)),
        Vector{T}(undef, 0),
        Vector{T}(undef, 0),
        Vector{W}(undef, 0)
    )
end

DensitySampleVector(::Type{S}, nparams::Integer) where {P<:Real,T<:AbstractFloat,W<:Real,S<:DensitySample{P,T,W}} =
    DensitySampleVector{P,T,W}(nparams)


Base.size(xs::DensitySampleVector) = size(xs.params)

Base.getindex(xs::DensitySampleVector, i::Integer) =
    DensitySample(xs.params[i], xs.log_posterior[i], xs.log_prior[i], xs.weight[i])

Base.@propagate_inbounds Base._getindex(l::IndexStyle, xs::DensitySampleVector, idxs::AbstractVector{<:Integer}) =
    DensitySampleVector(xs.params[:, idxs], xs.log_posterior[idxs], xs.log_prior[idxs], xs.weight[idxs])

Base.IndexStyle(xs::DensitySampleVector) = IndexStyle(xs.params)


function Base.push!(xs::DensitySampleVector, x::DensitySample)
    push!(xs.params, x.params)
    push!(xs.log_posterior, x.log_posterior)
    push!(xs.log_prior, x.log_prior)
    push!(xs.weight, x.weight)
    xs
end


function Base.append!(A::DensitySampleVector, B::DensitySampleVector)
    append!(A.params, B.params)
    append!(A.log_posterior, B.log_posterior)
    append!(A.log_prior, B.log_prior)
    append!(A.weight, B.weight)
    A
end


function Base.resize!(A::DensitySampleVector, n::Integer)
    resize!(A.params, n)
    resize!(A.log_posterior, n)
    resize!(A.log_prior, n)
    resize!(A.weight, n)
    A
end


Base.@propagate_inbounds function Base.unsafe_view(A::DensitySampleVector, idxs)
    DensitySampleVector(
        view(A.params, idxs),
        view(A.log_posterior, idxs),
        view(A.log_prior, idxs),
        view(A.weight, idxs)
    )
end


function UnsafeArrays.uview(A::DensitySampleVector)
    DensitySampleVector(
        uview(A.params),
        uview(A.log_posterior),
        uview(A.log_prior),
        uview(A.weight)
    )
end


Tables.istable(::Type{<:DensitySampleVector}) = true

Tables.columnaccess(::Type{<:DensitySampleVector}) = true

Tables.columns(A::DensitySampleVector) = (
    params = A.params,
    log_posterior = A.log_posterior,
    log_prior = A.log_prior,
    weight = A.weight
)

Tables.rowaccess(::Type{<:DensitySampleVector}) = true

Tables.rows(A::DensitySampleVector) = A

Tables.schema(A::DensitySampleVector) = Tables.Schema(
    (:params, :log_posterior, :log_prior, :weight),
    (eltype(A.params), eltype(A.log_posterior), eltype(A.log_prior), eltype(A.weight))
)


function _swap!(A::DensitySampleVector, i_A::SingleArrayIndex, B::DensitySampleVector, i_B::SingleArrayIndex)
    _swap!(view(flatview(A.params), :, i_A), view(flatview(A.params), :, i_B))  # Memory allocation!
    _swap!(A.log_posterior, i_A, B.log_posterior, i_B)
    _swap!(A.log_prior, i_A, B.log_prior, i_B)
    _swap!(A.weight, i_A, B.weight, i_B)
    A
end


function read_fom_hdf5(input, ::Type{DensitySampleVector})
    DensitySampleVector(
        VectorOfSimilarVectors(input["params"][:,:]),
        input["log_posterior"][:],
        input["log_prior"][:],
        input["weight"][:]
    )
end


function write_to_hdf5(output, samples::DensitySampleVector)
    output["params"] = Array(flatview(samples.params))
    output["log_posterior"] = samples.log_posterior
    output["log_prior"] = samples.log_posterior
    output["weight"] = samples.weight
    nothing
end
