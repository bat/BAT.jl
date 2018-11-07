# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Base: @propagate_inbounds


@doc """
    OnlineUvMean{T<:AbstractFloat}

Univariate mean implemented via Kahan-Babuška-Neumaier summation.
"""
mutable struct OnlineUvMean{T<:AbstractFloat}
    sum_v::DoubleFloat{T}
    sum_w::DoubleFloat{T}

    OnlineUvMean{T}() where {T<:AbstractFloat} = new{T}(zero(DoubleFloat{T}), zero(DoubleFloat{T}))

    OnlineUvMean{T}(sum_v::Real, sum_w::Real) where {T<:AbstractFloat} = new{T}(sum_v, sum_w)
end

export OnlineUvMean

OnlineUvMean() = OnlineUvMean{Float64}()

@inline Base.getindex(omn::OnlineUvMean{T}) where {T<:AbstractFloat} = T(omn.sum_v / omn.sum_w)


function Base.merge!(target::OnlineUvMean{T}, others::OnlineUvMean...) where {T}
    sum_v = target.sum_v
    sum_w = target.sum_w

    @inbounds @simd for x in others
        sum_w += x.sum_w
        sum_v += x.sum_v
    end

    target.sum_w = sum_w
    target.sum_v = sum_v

    target
end


@inline function _push_impl!(omn::OnlineUvMean{T}, data::Real, weight::Real) where {T}
    # Workaround for lack of promotion between, e.g., Float32 and DoubleFloat{Float64}
    weight_conv = T(weight)

    omn.sum_v += DoubleFloat{T}(weight_conv * data)
    omn.sum_w += DoubleFloat{T}(weight_conv)
    omn
end



@doc """
    OnlineUvVar{T<:AbstractFloat,W}

Implementation based on variance calculation Algorithms of Welford and West.

`W` must either be `Weights` (no bias correction) or one of `AnalyticWeights`,
`FrequencyWeights` or `ProbabilityWeights` to specify the desired bias
correction method.
"""

mutable struct OnlineUvVar{T<:AbstractFloat,W}
    n::Int64
    sum_w::DoubleFloat{T}
    sum_w2::DoubleFloat{T}
    mean_x::T
    s::T

    OnlineUvVar{T,W}() where {T<:AbstractFloat,W} =
        new{T,W}(
            zero(Int64), zero(DoubleFloat{T}), zero(DoubleFloat{T}),
            zero(T), zero(T)
        )

    OnlineUvVar{T,W}(n::Int64, sum_w::DoubleFloat{T}, sum_w2::DoubleFloat{T}, mean_x::T, s::T) where {T<:AbstractFloat,W} =
        new{T,W}(
            n, sum_w, sum_w2, mean_x, s
        )
end

export OnlineUvVar

OnlineUvVar() = OnlineUvVar{Float64, ProbabilityWeights}()


@propagate_inbounds Base.getindex(ocv::OnlineUvVar{T, Weights}) where {T} =
    ifelse(ocv.sum_w > zero(T), T(ocv.s / ocv.sum_w), T(NaN))

@propagate_inbounds function Base.getindex(ocv::OnlineUvVar{T, AnalyticWeights}) where {T}
    d = ocv.sum_w - ocv.sum_w2 / ocv.sum_w
    ifelse(ocv.sum_w > zero(T) && d > zero(T), T(ocv.s / d), T(NaN))
end

@propagate_inbounds Base.getindex(ocv::OnlineUvVar{T, FrequencyWeights}) where {T} =
    ifelse(ocv.sum_w > one(T), T(ocv.s / (ocv.sum_w - one(T))), T(NaN))


@propagate_inbounds Base.getindex(ocv::OnlineUvVar{T, ProbabilityWeights}) where {T} =
    ifelse(ocv.n > one(T) && ocv.sum_w > zero(T), T(ocv.s * ocv.n / ((ocv.n - one(T)) * ocv.sum_w)), T(NaN))



function Base.merge!(target::OnlineUvVar{T,W}, others::OnlineUvVar...) where {T,W}
    n = target.n
    sum_w = target.sum_w
    sum_w2 = target.sum_w2
    mean_x = target.mean_x
    s = target.s

    @inbounds @simd for x in others
        n += x.n

        dx = mean_x - x.mean_x

        new_sum_w = (sum_w + x.sum_w)
        mean_x = (sum_w * mean_x + x.sum_w * x.mean_x) / new_sum_w

        s += x.s + sum_w * x.sum_w / new_sum_w * dx * dx

        sum_w = new_sum_w
        sum_w2 += x.sum_w2

    end

    target.n = n
    target.sum_w = sum_w
    target.sum_w2 = sum_w2
    target.mean_x = mean_x
    target.s = s

    target
end


@inline function _push_impl!(ocv::OnlineUvVar{T}, data::Real, weight::Real) where T
    # Ignore zero weights (can't be handled)
    if weight ≈ 0
        return ocv
    end

    # Workaround for lack of promotion between, e.g., Float32 and DoubleFloat{Float64}
    weight_conv = T(weight)

    n = ocv.n
    sum_w = ocv.sum_w
    sum_w2 = ocv.sum_w2
    mean_x = ocv.mean_x
    s = ocv.s

    n += one(n)
    sum_w += DoubleFloat{T}(weight_conv)
    sum_w2 += DoubleFloat{T}(weight_conv^2)
    dx = data - mean_x
    new_mean_x = mean_x + dx * weight_conv / sum_w
    new_dx = data - new_mean_x

    s = muladd(dx, weight_conv * new_dx, s)
    mean_x = new_mean_x

    ocv.n = n
    ocv.sum_w = sum_w
    ocv.sum_w2 = sum_w2
    ocv.mean_x = mean_x
    ocv.s = s

    ocv
end



mutable struct BasicUvStatistics{T<:Real,W}
    mean::OnlineUvMean{T}
    var::OnlineUvVar{T,W}
    maximum::T
    minimum::T

    BasicUvStatistics{T,W}() where {T<:Real,W} =
        new(OnlineUvMean{T}(), OnlineUvVar{T,W}(), typemin(T), typemax(T))
    BasicUvStatistics{T,W}(mean::OnlineUvMean{T}, var::OnlineUvVar{T,W}, maximum::T, minimum::T) where {T<:Real,W} =
        new(mean, var, maximum, minimum)
end

export BasicUvStatistics



@inline function _push_impl!(stats::BasicUvStatistics{T,W}, data::Real, weight::Real) where {T,W}
    nmaximum = stats.maximum
    nminimum = stats.minimum

    push!(stats.mean, data, weight)
    push!(stats.var, data, weight)
    nmaximum = max(stats.maximum, maximum(data))
    nminimum = min(stats.minimum, minimum(data))

    stats.maximum = nmaximum
    stats.minimum = nminimum

    stats
end

function Base.merge!(target::BasicUvStatistics, others::BasicUvStatistics...)
    t_mean = target.mean
    t_var = target.var
    t_maximum = target.maximum
    t_minimum = target.minimum

    for x in others
        t_mean = merge!(t_mean, x.mean)
        t_var = merge!(t_var, x.var)
        t_maximum = max(t_maximum, x.maximum)
        t_minimum = min(t_minimum, x.minimum)
    end

    target.mean = t_mean
    target.var = t_var
    target.maximum = t_maximum
    target.minimum = t_minimum

    target
end



const OnlineUvStatistic = Union{BAT.OnlineUvMean, BAT.OnlineUvVar, BAT.BasicUvStatistics}


Base.push!(ocv::OnlineUvStatistic, data::Real, weight::Real = 1) =
    _push_impl!(ocv, data, weight)


@inline function Base.append!(stats::OnlineUvStatistic, data::AbstractVector{<:Real})
    @inbounds for i in axes(data, 1)
        push!(stats, data[i])
    end
    stats
end

@inline function Base.append!(stats::OnlineUvStatistic, data::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
    @assert axes(data) == axes(weights)# ToDo: Throw exception instead of assert
    @inbounds for i in axes(data, 1)
        push!(stats, data[i], weights[i])
    end
    stats
end


Base.merge(target::S, others::S...) where{S <: OnlineUvStatistic} = merge!(deepcopy(target), others...)
