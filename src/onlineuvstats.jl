# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Base.@propagate_inbounds
using StatsBase
using DoubleDouble


"""
    OnlineUvMean{T<:AbstractFloat}

Multi-variate mean implemented via Kahan-Babuška-Neumaier summation.
"""
mutable struct OnlineUvMean{T<:AbstractFloat}
    sum_v::Double{T}
    sum_w::Double{T}

    OnlineUvMean{T}() where {T<:AbstractFloat} = new{T}(zero(Double{T}), zero(Double{T}))

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


@inline function push_contiguous!(omn::OnlineUvMean{T}, data, weight::Array{<:Real, 1}) where {T}
    @inbounds @simd for i in indices(data, 1)
        push_contiguous!(omn, data[i], weight[i])
    end
    omn
end

@inline function push_contiguous!(omn::OnlineUvMean{T}, data::T, weight::T) where {T<:Real} 
    omn.sum_v += Single(weight*data)
    omn.sum_w += Single(weight)
    omn
end



"""
    OnlineUvVar{T<:AbstractFloat,W}

Implementation based on variance calculation Algorithms of Welford and West.

`W` must either be `Weights` (no bias correction) or one of `AnalyticWeights`,
`FrequencyWeights` or `ProbabilityWeights` to specify the desired bias
correction method.
"""

mutable struct OnlineUvVar{T<:AbstractFloat,W}
    n::Int64
    sum_w::Double{T}
    sum_w2::Double{T}
    mean_x::T
    s::T

    OnlineUvVar{T,W}() where {T<:AbstractFloat,W} =
        new{T,W}(
            zero(Int64), zero(Double{T}), zero(Double{T}),
            zero(T), zero(T)
        )
    
    OnlineUvVar{T,W}(n::Int64, sum_w::Double{T}, sum_w2::Double{T}, mean_x::T, s::T) where {T<:AbstractFloat,W} =
        new{T,W}(
            n, sum_w, sum_w2, mean_x, s
        )    
end

export OnlineUvVar

OnlineUvVar() = OnlineUvVar{Float64, ProbabilityWeights}()


@propagate_inbounds Base.getindex{T}(ocv::OnlineUvVar{T, Weights}) =
    ifelse(ocv.sum_w > 0, T(ocv.s / ocv.sum_w), T(NaN))

@propagate_inbounds function Base.getindex{T}(ocv::OnlineUvVar{T, AnalyticWeights}) 
    d = ocv.sum_w - ocv.sum_w2 / ocv.sum_w
    ifelse(ocv.sum_w > 0 && d > 0, T(ocv.s / d), T(NaN))
end

@propagate_inbounds Base.getindex{T}(ocv::OnlineUvVar{T, FrequencyWeights}) =
    ifelse(ocv.sum_w > 1, T(ocv.s / (ocv.sum_w - 1)), T(NaN))    


@propagate_inbounds Base.getindex{T}(ocv::OnlineUvVar{T, ProbabilityWeights}) =
    ifelse(ocv.n > 1 && ocv.sum_w > 0, T(ocv.s * ocv.n / ((ocv.n - 1) * ocv.sum_w)), T(NaN))



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



@inline function push_contiguous!{T,W}(ocv::OnlineUvVar{T,W}, data, weight::Array{<:Real, 1})
    @inbounds for i in indices(data, 1)
        ocv = push_contiguous!(ocv, data[i], weight[i])
    end
    ocv
end

@inline function push_contiguous!{T,W}(ocv::OnlineUvVar{T,W}, data::Real, weight::Real)
    n = ocv.n
    sum_w = ocv.sum_w
    sum_w2 = ocv.sum_w2
    mean_x = ocv.mean_x
    s = ocv.s

    n += one(n)
    sum_w += Single(weight)
    sum_w2 += Single(weight^2)
    dx = data - mean_x
    new_mean_x = mean_x + dx * weight / sum_w
    new_dx = data - new_mean_x
    
    s = muladd(dx, weight * new_dx, s)
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


@inline function push_contiguous!{T,W}(stats::BasicUvStatistics{T,W}, data, weight::Array{<:Real, 1})
    @inbounds for i in indices(data, 1)
        stats = push_contiguous!(stats, data[i], weight[i])
    end
    stats
end

@inline function push_contiguous!{T,W}(stats::BasicUvStatistics{T,W}, data::Real, weight::Real = one(T))
    new_mean = push!(stats.mean, data, weight)
    new_var = push!(stats.var, data, weight)
    new_maximum = max(stats.maximum, maximum(data))
    new_minimum = min(stats.minimum, minimum(data))
    BasicUvStatistics{T,W}(new_mean, new_var, new_maximum, new_minimum)
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

const OnlineUvStatistic{T, W} = Union{BAT.OnlineUvMean{T}, BAT.OnlineUvVar{T, W}, BAT.BasicUvStatistics{T, W}} where W where T

Base.push!(ocv::OnlineUvStatistic{T}, data::T, weight::T = one(T)) where T =
    push_contiguous!(ocv, data, weight)

Base.push!(ocv::OnlineUvStatistic{T}, data::NTuple{N, T}, weight::Array{T, 1}=ones(T, N)) where{T, N} =
    push_contiguous!(ocv, collect(data), weight)

Base.push!(ocv::OnlineUvStatistic{T}, data::Array{T, 1}, weight::Array{T, 1})  where T =
    push_contiguous!(ocv, data, weight)

Base.push!(ocv::OnlineUvStatistic{T}, data::Array{T, 1}) where T = 
    push_contiguous!(ocv, data, ones(T, size(data, 1)))

Base.merge(target::S, others::S...) where{S <: OnlineUvStatistic} = merge!(deepcopy(target), others...)
