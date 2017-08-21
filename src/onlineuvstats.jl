# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using StatsBase
using DoubleDouble


"""
    OnlineMean{T<:AbstractFloat} <: AbstractVector{T}

Multi-variate mean implemented via Kahan-Babuška-Neumaier summation.
"""
struct OnlineMean{T<:AbstractFloat} <: AbstractVector{T}
    sum_v::Double{T}
    sum_w::Double{T}

    OnlineMean{T}() where {T<:AbstractFloat} = new{T}(zero(Double{T}), zero(Double{T}))

    OnlineMean{T}(sum_v::Real, sum_w::Real) where {T<:AbstractFloat} = new{T}(sum_v, sum_w)
end

export OnlineMean

OnlineMean() = OnlineUvMean{Float64}()

@inline Base.getindex(omn::OnlineMean{T}) where {T} = T(omn.sum_v / omn.sum_w)


function Base.merge!(target::OnlineMean{T}, others::OnlineMean...) where {T}
    sum_v = target.sum_v
    sum_w = target.sum_w

    @inbounds @simd for x in others
        sum_w += x.sum_w
        sum_v += x.sum_v
    end

    OnlineMean{T}(sum_v, sum_w)
end


@inline function _append_impl(omn::OnlineMean{T}, data, weight::Real = one(T)) where {T}
    @inbounds @simd for x in data
        omn = OnlineMean{T}(sum_v + Single(x), sum_w + Single(weight))
    end
    omn
end

Base.append(omn::OnlineMean{T}, data::NTuple(N, <:Real), weight::Real = one(T)) where {T,N} =
    _append_impl(omn, data, weight)

Base.append(omn::OnlineMean{T}, data::AbstractArray{<:Real}, weight::Real = one(T)) where {T,N} =
    _append_impl(omn, data, weight)



"""
    OnlineUvVar{T<:AbstractFloat,W} <: AbstractMatrix{T}

Implementation based on variance calculation Algorithms of Welford and West.

`W` must either be `Weights` (no bias correction) or one of `AnalyticWeights`,
`FrequencyWeights` or `ProbabilityWeights` to specify the desired bias
correction method.
"""

struct OnlineUvVar{T<:AbstractFloat,W} <: AbstractMatrix{T}
    n::Int64
    sum_w::Double{T}
    sum_w2::Double{T}
    mean_x::T
    s::T

    OnlineUvVar{T,W}(m::Integer) where {T<:AbstractFloat,W} =
        new{T,W}(
            m, zero(Int64), zero(Double{T}), zero(Double{T}),
            zeros(T, m), zeros(T, m), zeros(T, m, m)
        )
end

export OnlineUvVar

OnlineUvVar(m::Integer) = OnlineUvVar{Float64, ProbabilityWeights}(m::Integer)


@propagate_inbounds Base.getindex{T}(ocv::OnlineUvVar{T, Weights}) =
    ifelse(ocv.sum_w > 0, T(ocv.s / ocv.sum_w), T(NaN))

@propagate_inbounds Base.getindex{T}(ocv::OnlineUvVar{T, AnalyticWeights}) =
    ifelse(ocv.sum_w > 1, T(ocv.s / (ocv.sum_w - 1)), T(NaN))

@propagate_inbounds function Base.getindex{T}(ocv::OnlineUvVar{T, FrequencyWeights})
    d = ocv.sum_w - ocv.sum_w2 / ocv.sum_w
    ifelse(ocv.sum_w > 0 && d > 0, T(ocv.s / d), T(NaN))
end

@propagate_inbounds Base.getindex{T}(ocv::OnlineUvVar{T, ProbabilityWeights}) =
    ifelse(ocv.n > 1 && ocv.sum_w > 0, T(ocv.s * ocv.n / ((ocv.n - 1) * ocv.sum_w)), T(NaN))



function Base.merge(target::OnlineUvVar{T,W}, others::OnlineUvVar...) where {T,W}
    n = target.n
    sum_w = target.sum_w
    sum_w2 = target.sum_w2
    mean_x = target.mean_x
    s = target.s

    @inbounds @simd for x in others
        target.n = x.n
        target.sum_w = x.sum_w
        target.sum_w2 = x.sum_w2
        target.mean_x = x.mean_x
        target.s = x.s
    end

    OnlineUvVar{T,W}(n, sum_w, sum_w2, mean_x, s)
end



@inline function _append_impl{T,W}(ocv::OnlineUvVar{T,W}, data, weight::Real = one(T))
    n = ocv.n
    sum_w = ocv.sum_w
    sum_w2 = ocv.sum_w2
    mean_x = ocv.mean_x
    s = ocv.s

    n += one(n)
    sum_w += Single(weight)
    sum_w2 += Single(weight^2)
    dx = x - mean_x
    new_mean_x = mean_x + dx / sum_w
    new_dx = x - new_mean_x
    ocv.s = muladd(dx, new_dx, s)
    ocv.mean_x = new_mean_x

    OnlineUvVar{T,W}(n, sum_w, sum_w2, mean_x, s)
end

Base.append(ocv::OnlineUvVar{T,W}, data::NTuple(N, <:Real), weight::Real = one(T)) where {T,W,N} =
    _append_impl(ocv, data, weight)

Base.append(ocv::OnlineUvVar{T,W}, data::AbstractArray{<:Real}, weight::Real = one(T)) where {T,W} =
    _append_impl(ocv, data, weight)



mutable struct BasicUvStatistics{T,W}
    mean::OnlineUvMean{T}
    var::OnlineUvVar{T,W}
    maximum{T}
    minimum{T}
end

export BasicUvStatistics


@inline function _append_impl{T,W}(stats::BasicUvStatistics{T,W}, data, weight::Real = one(T))
    new_mean = append(stats.mean, data, weight)
    new_var = append(stats.var, data, weight)
    new_maximum = max(stats.maximum, maximum(data))
    new_minimum = min(stats.minimum, minimum(data))
    BasicUvStatistics{T,W}(new_mean, new_var, new_maximum, new_minimum)
end

Base.append(stats::BasicUvStatistics{T,W}, data::NTuple(N, <:Real), weight::Real = one(T)) where {T,W,N} =
    _append_impl(stats, data, weight)

Base.append(stats::BasicUvStatistics{T,W}, data::AbstractArray{<:Real}, weight::Real = one(T)) where {T,W} =
    _append_impl(stats, data, weight)


function Base.merge!(target::BasicUvStatistics, others::BasicUvStatistics...)
    t_mean = target.mean
    t_var = target.var
    t_maximum = target.maximum
    t_minimum = target.minimum

    for x in others
        t_mean = merge(t_mean, x.mean)
        t_var = merge(t_var, x.var)
        t_maximum = merge(t_maximum, x.maximum)
        t_minimum = merge(t_minimum, x.minimum)
    end
    target
end
