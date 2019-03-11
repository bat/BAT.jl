# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Base: @propagate_inbounds


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



@doc """
    OnlineMvMean{T<:AbstractFloat} <: AbstractVector{T}

Multi-variate mean implemented via Kahan-Babuška-Neumaier summation.
"""
mutable struct OnlineMvMean{T<:AbstractFloat} <: AbstractVector{T}
    m::Int
    sum_w::DoubleFloat{T}
    S::Vector{T}
    C::Vector{T}

    OnlineMvMean{T}(m::Integer) where {T<:AbstractFloat} =
        new{T}(m, zero(DoubleFloat{T}), zeros(T, m), zeros(T, m))
end

export OnlineMvMean

OnlineMvMean(m::Integer) = OnlineMvMean{Float64}(m::Integer)


@inline Base.size(omn::OnlineMvMean) = size(omn.S)

@propagate_inbounds function Base.getindex(omn::OnlineMvMean{T}, idxs::Integer...) where {T}
    T((DoubleFloat{T}(omn.S[idxs...]) + DoubleFloat{T}(omn.C[idxs...])) / omn.sum_w)
end


function Base.merge!(target::OnlineMvMean, others::OnlineMvMean...)
    for x in others
        target.m != x.m && throw(ArgumentError("can't merge OnlineMvMean instances with different size"))
    end
    for x in others
        target.sum_w += x.sum_w
        target_S = target.S; target_C = target.C
        x_S = x.S; x_C = x.C
        @assert eachindex(target_S) == eachindex(x.S) == eachindex(target_C) == eachindex(x.C)  # TODO: Use exception instead of assert
        @inbounds @simd for i in eachindex(target_S)
            target_S[i], target_C[i] = kbn_add(
                (target_S[i], target_C[i]),
                (x.S[i], x.C[i])
            )
        end
    end
    target
end

@inline function Base.push!(
    omn::OnlineMvMean{T}, data::AbstractVector{<:Real}, weight::Real = one(T)
) where {T}
    # Workaround for lack of promotion between, e.g., Float32 and DoubleFloat{Float64}
    weight_conv = T(weight)

    m = omn.m
    S = omn.S
    C = omn.C

    idxs = axes(data, 1)
    @assert idxs == axes(S, 1) == axes(C, 1)  # TODO: Use exception instead of assert

    omn.sum_w += weight_conv

    @inbounds @simd for i in idxs
        x = weight_conv * data[i]
        S[i], C[i] = kbn_add((S[i], C[i]), x)
    end

    omn
end



@doc """
    OnlineMvCov{T<:AbstractFloat,W} <: AbstractMatrix{T}

Implementation based on variance calculation Algorithms of Welford and West.

`W` must either be `Weights` (no bias correction) or one of `AnalyticWeights`,
`FrequencyWeights` or `ProbabilityWeights` to specify the desired bias
correction method.
"""

mutable struct OnlineMvCov{T<:AbstractFloat,W} <: AbstractMatrix{T}
    m::Int
    n::Int64
    sum_w::DoubleFloat{T}
    sum_w2::DoubleFloat{T}
    Mean_X::Vector{T}
    New_Mean_X::Vector{T}
    S::Matrix{T}

    OnlineMvCov{T,W}(m::Integer) where {T<:AbstractFloat,W} =
        new{T,W}(
            m, zero(Int64), zero(DoubleFloat{T}), zero(DoubleFloat{T}),
            zeros(T, m), zeros(T, m), zeros(T, m, m)
        )
end

export OnlineMvCov

OnlineMvCov(m::Integer) = OnlineMvCov{Float64, ProbabilityWeights}(m::Integer)

@inline Base.size(ocv::OnlineMvCov) = size(ocv.S)

@propagate_inbounds function Base.getindex(ocv::OnlineMvCov{T, Weights}, idxs::Integer...) where {T}
    sum_w = ocv.sum_w
    ifelse(
        sum_w > zero(T),
        T(ocv.S[idxs...] / sum_w),
        T(NaN)
    )
end

@propagate_inbounds function Base.getindex(ocv::OnlineMvCov{T, AnalyticWeights}, idxs::Integer...) where {T}
    sum_w = ocv.sum_w
    sum_w2 = ocv.sum_w2
    d = sum_w - sum_w2 / sum_w
    ifelse(
        sum_w > zero(T) && d > zero(T),
        T(ocv.S[idxs...] / d),
        T(NaN)
    )
end

@propagate_inbounds function Base.getindex(ocv::OnlineMvCov{T, FrequencyWeights}, idxs::Integer...) where {T}
    sum_w = ocv.sum_w
    ifelse(
        sum_w > one(T),
        T(ocv.S[idxs...] / (sum_w - one(T))),
        T(NaN)
    )
end

@propagate_inbounds function Base.getindex(ocv::OnlineMvCov{T, ProbabilityWeights}, idxs::Integer...) where {T}
    n = ocv.n
    sum_w = ocv.sum_w
    ifelse(
        n > one(T) && sum_w > zero(T),
        T(ocv.S[idxs...] * n / ((n - one(T)) * sum_w)),
        T(NaN)
    )
end


function Base.merge!(target::OnlineMvCov{T,W}, others::OnlineMvCov{T,W}...) where {T,W}
    # TODO: Improve implementation

    for x in others
        target.m != x.m && throw(ArgumentError("can't merge OnlineMvCov instances with different size"))
    end
    m = target.m
    for x in others
        idxs = Base.OneTo(m)
        scl = (target.sum_w * x.sum_w) / (target.sum_w + x.sum_w)
        for j in idxs
            for i in idxs
                target.S[i, j] = scl *
                    (target.Mean_X[i] - x.Mean_X[i]) *
                    (target.Mean_X[j] - x.Mean_X[j]) +
                    target.S[i, j] + x.S[i, j]
            end
        end

        d = inv(T(target.sum_w + x.sum_w))
        for i in idxs
            target.Mean_X[i] = (target.Mean_X[i] * target.sum_w + x.Mean_X[i] * x.sum_w) * DoubleFloat(d)
            target.New_Mean_X[i] = target.Mean_X[i]
        end

        target.n += x.n
        target.sum_w += x.sum_w
        target.sum_w2 += x.sum_w2
    end
    target
end



@inline function Base.push!(
    ocv::OnlineMvCov{T,W}, data::AbstractVector{<:Real}, weight::Real = one(T)
) where {T,W}
    # Ignore zero weights (can't be handled)
    if weight ≈ 0
        return ocv
    end

    # Workaround for lack of promotion between, e.g., Float32 and DoubleFloat{Float64}
    weight_conv = T(weight)

    m = ocv.m
    n = ocv.n
    sum_w = ocv.sum_w
    sum_w2 = ocv.sum_w2
    Mean_X = ocv.Mean_X
    New_Mean_X = ocv.New_Mean_X
    S = ocv.S

    idxs = axes(data, 1)
    @assert idxs == axes(Mean_X, 1) == axes(New_Mean_X, 1) == axes(S, 1) == axes(S, 2)  # TODO: Use exception instead of assert

    n += one(n)
    sum_w += weight_conv
    sum_w2 += weight_conv^2

    weight_over_sum_w = T(weight_conv / sum_w)

    @inbounds @simd for i in idxs
        x = data[i]
        mean_X = Mean_X[i]
        New_Mean_X[i] = muladd(x - mean_X, weight_over_sum_w, mean_X)
    end

    @inbounds for j in idxs
        new_dx_j = data[j] - New_Mean_X[j]

        j_offs = (LinearIndices(S))[last(idxs), j] - last(idxs)
        #@assert (LinearIndices((S))[last(idxs), j] == last(idxs) + j_offs  # TODO: Use exception instead of assert
        @simd for i in idxs
            dx_i = data[i] - Mean_X[i]
            S[i + j_offs] = muladd(dx_i, weight_conv * new_dx_j, S[i + j_offs])
        end
    end

    @inbounds @simd for i in idxs
        Mean_X[i] = New_Mean_X[i]
    end

    ocv.n = n
    ocv.sum_w = sum_w
    ocv.sum_w2 = sum_w2

    ocv
end

@doc """
    BasicMvStatistics{T<:Real,W}

`W` must either be `Weights` (no bias correction) or one of `AnalyticWeights`,
`FrequencyWeights` or `ProbabilityWeights` to specify the desired bias
correction method.
"""


mutable struct BasicMvStatistics{T<:Real,W}
    m::Int
    mean::OnlineMvMean{T}
    cov::OnlineMvCov{T,W}
    maximum::Vector{T}
    minimum::Vector{T}

    BasicMvStatistics{T,W}(
        mean::OnlineMvMean{T},
        cov::OnlineMvCov{T,W},
        maximum::Vector{T},
        minimum::Vector{T}
    ) where {T<:Real,W} = new(mean, cov, maximum, minimum)

    BasicMvStatistics{T,W}(m::Integer) where {T<:Real,W} =
        new(m, OnlineMvMean{T}(m), OnlineMvCov{T,W}(m), fill(typemin(T), m), fill(typemax(T), m))
end

export BasicMvStatistics


function Base.merge!(target::BasicMvStatistics, others::BasicMvStatistics...)
    for x in others
        merge!(target.mean, x.mean)
        merge!(target.cov, x.cov)
        target.maximum = max.(target.maximum, x.maximum)
        target.minimum = min.(target.minimum, x.minimum)
    end
    target
end


function Base.push!(
    stats::BasicMvStatistics{T,W}, data::AbstractVector{<:Real}, weight::Real = one(T)
) where {T,W}
    push!(stats.mean, data, weight)
    push!(stats.cov, data, weight)

    max_v = stats.maximum
    min_v = stats.minimum

    idxs = axes(data, 1)
    @assert idxs == axes(max_v, 1) == axes(min_v, 1)  # TODO: Use exception instead of assert

    @inbounds @simd for i in idxs
        x = data[i]
        max_v[i] = max(max_v[i], x)
        min_v[i] = min(min_v[i], x)
    end

    stats
end



const OnlineMvStatistic = Union{OnlineMvMean, OnlineMvCov, BasicMvStatistics}


function Base.append!(target::OnlineMvStatistic, data::VectorOfSimilarVectors)
    @uviews data push!.(Scalar(target), data)
    target
end


function Base.append!(target::OnlineMvStatistic, data::VectorOfSimilarVectors, weights::AbstractVector)
    @uviews data weights push!.(Scalar(target), data, weights)
    target
end


Base.merge(x::S, others::S...) where {S <: OnlineMvStatistic} = merge!(deepcopy(x), others...)
