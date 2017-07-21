# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Base.@propagate_inbounds
using DoubleDouble


# SIMD-compatible KBN-summation

@inline function kbn_add(a::NTuple{2, Real}, b::Real)
    s = a[1] + b
    c = a[2] + ifelse(
        abs(a[1]) >= abs(b),
        (a[1] - s) + b,
        (b - s) + a[1]
    )
    (s, c)
end

@inline function kbn_add(a::NTuple{2, Real}, b::NTuple{2, Real})
    s = a[1] + b[1]
    c = ifelse(
        abs(a[1]) > abs(b[1]),
        (((a[1] - s) + b[1]) + b[2]) + a[2],
        (((b[1] - s) + a[1]) + a[2]) + b[2]
    )
    (s, c)
end



"""
    OnlineMvMean{T<:AbstractFloat} <: AbstractVector{T}

Multi-variate mean implemented via Kahan-Babuška-Neumaier summation.
"""
mutable struct OnlineMvMean{T<:AbstractFloat} <: AbstractVector{T}
    m::Int
    wsum::Double{T}
    S::Vector{T}
    C::Vector{T}

    OnlineMvMean{T}(m::Integer) where {T<:AbstractFloat} =
        new{T}(m, zero(Int64), zeros(T, m), zeros(T, m))
end

export OnlineMvMean

OnlineMvMean(m::Integer) = OnlineMvMean{Float64}(m::Integer)


@inline Base.size(omn::OnlineMvMean) = size(omn.S)

@propagate_inbounds function Base.getindex{T}(omn::OnlineMvMean{T}, idxs::Integer...)
    T((Single(omn.S[idxs...]) + Single(omn.C[idxs...])) / omn.wsum)
end



@inline Base.push!(omn::OnlineMvMean, data::Vector, weight::Real = one(T)) =
    push_contiguous!(omn, data, first(linearindices(data)), weight)


function Base.append!(omn::OnlineMvMean, data::Matrix, vardim::Integer = 1)
    if (vardim == 1)
        throw(ArgumentError("vardim == $vardim not supported (yet)"))
    elseif (vardim == 2)
        @assert omn.m == size(data, 1)  # TODO: Replace by exception
        @inbounds for i in indices(data, 2)
            push_contiguous!(omn, data, sub2ind(data, 1, i))
        end
    else
        throw(ArgumentError("Value of vardim must be 2, not $vardim"))
    end

    omn
end


function Base.append!(omn::OnlineMvMean, data::Matrix, weights::Vector, vardim::Integer = 1)
    if (vardim == 1)
        throw(ArgumentError("vardim == $vardim not supported (yet)"))
    elseif (vardim == 2)
        @assert omn.m == size(data, 1)  # TODO: Replace by exception
        @assert size(data, 2) == size(weights, 1)  # TODO: Replace by exception
        @inbounds for i in indices(data, 2)
            push_contiguous!(omn, data, sub2ind(data, 1, i), weights[i])
        end
    else
        throw(ArgumentError("Value of vardim must be 2, not $vardim"))
    end

    omn
end


function Base.merge!(target::OnlineMvMean, others::OnlineMvMean...)
    for x in others
        target.m != x.m && throw(ArgumentError("can't merge OnlineMvMeans with different size"))
    end
    for x in others
        target.wsum += x.wsum
        target_S = target.S; target_C = target.C
        x_S = x.S; x_C = x.C
        @assert eachindex(target_S) == eachindex(x.S) == eachindex(target_C) == eachindex(x.C)
        @inbounds @simd for i in eachindex(target_S)
            target_S[i], target_C[i] = kbn_add(
                (target_S[i], target_C[i]),
                (x.S[i], x.C[i])
            )
        end
    end
    target
end

Base.merge(x::OnlineMvMean, others::OnlineMvMean...) = merge!(deepcopy(x), others...)



@inline function push_contiguous!{T}(
    omn::OnlineMvMean{T}, data::Array,
    start::Integer, weight::Real = one(T)
)
    m = omn.m
    S = omn.S
    C = omn.C

    idxs = Base.OneTo(m)

    dshft = Int(start) - 1

    @boundscheck begin
        @assert idxs == indices(S, 1) == indices(C, 1)
        checkbounds(data, idxs + dshft)
    end

    
    @inbounds @simd for i in idxs
        x = weight * data[i + dshft]
        S[i], C[i] = kbn_add((S[i], C[i]), x)
    end

    omn.wsum += Single(weight)

    omn
end



"""
    OnlineMvCov{T<:AbstractFloat} <: AbstractMatrix{T}

Implementation based on variance calculation Algorithms of Welford, West
and Chan et al. .
"""

mutable struct OnlineMvCov{T<:AbstractFloat} <: AbstractMatrix{T}
    m::Int
    n::Int64
    corrected::Bool
    Mean_X::Vector{T}
    New_Mean_X::Vector{T}
    S::Matrix{T}

    OnlineMvCov{T}(m::Integer, corrected::Bool = true) where {T<:AbstractFloat} =
        new{T}(m, zero(Int64), corrected, zeros(T, m), zeros(T, m), zeros(T, m, m))
end

export OnlineMvCov

OnlineMvCov(m::Integer, corrected::Bool = true) = OnlineMvCov{Float64}(m::Integer, corrected)


@inline Base.size(ocv::OnlineMvCov) = size(ocv.S)

@propagate_inbounds function Base.getindex(ocv::OnlineMvCov, idxs::Integer...)
    n_corr = ocv.n - Int(ocv.corrected)
    n_corr_clamped = ifelse(n_corr >= 0, n_corr, 0)
    ocv.S[idxs...] / n_corr_clamped
end



@inline Base.push!(ocv::OnlineMvCov, data::Vector) =
    push_contiguous!(ocv, data, first(linearindices(data)))


@inline function Base.append!(ocv::OnlineMvCov, data::Matrix, vardim::Integer = 1)
    if (vardim == 1)
        throw(ArgumentError("vardim == $vardim not supported (yet)"))
    elseif (vardim == 2)
        @assert(ocv.m == size(data, 1))
        @inbounds for i in indices(data, 2)
            push_contiguous!(ocv, data, sub2ind(data, 1, i))
        end
    else
        throw(ArgumentError("Value of vardim must be 2, not $vardim"))
    end

    ocv
end


# function Base.append!(a::OnlineMvCov, b::OnlineMvCov)
#     ...
# end


@inline function push_contiguous!{T}(
    ocv::OnlineMvCov{T}, data::Array,
    start::Integer
)
    m = ocv.m
    n = ocv.n
    Mean_X = ocv.Mean_X
    New_Mean_X = ocv.New_Mean_X
    S = ocv.S

    idxs = Base.OneTo(m)

    dshft = Int(start) - 1

    @boundscheck begin
        @assert idxs == indices(Mean_X, 1) == indices(New_Mean_X, 1) == indices(S, 1) == indices(S, 2)
        checkbounds(data, idxs + dshft)
    end

    n += 1
    n_inv = inv(T(n))

    @inbounds @simd for i in idxs
        x = data[i + dshft]
        mean_X = Mean_X[i]
        New_Mean_X[i] = muladd(x - mean_X, n_inv, mean_X)
    end

    @inbounds for j in idxs
        new_dx_j = data[j + dshft] - New_Mean_X[j]
        j_offs = sub2ind(S, 0, j)
        @assert sub2ind(S, last(idxs), j) == last(idxs) + j_offs
        @simd for i in idxs
            dx_i = data[i + dshft] - Mean_X[i]
            S[i + j_offs] += dx_i * new_dx_j
        end
    end

    @inbounds @simd for i in idxs
        Mean_X[i] = New_Mean_X[i]
    end

    ocv.n = n

    ocv
end
