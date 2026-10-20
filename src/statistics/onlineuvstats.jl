# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using Base: @propagate_inbounds


# Reweighting is a two-pass operation: all numeric conversions/range checks
# finish before the first write, including for composite statistics. Callers
# must own the statistics exclusively (this is not an interruption transaction).
function _reweight_relative!(target, factor::Real)
    get_zero_subnormals() && throw(ArgumentError("reweighting requires gradual underflow (subnormal flushing must be disabled)"))
    isfinite(factor) && factor >= 0 || throw(ArgumentError("reweighting factor must be finite and nonnegative"))
    iszero(factor) && return empty!(target)
    isone(factor) && return target
    _check_reweight(target, factor)
    _apply_reweight!(target, factor)
    target
end

function _reweight_factor(::Type{T}, factor::Real) where {T<:AbstractFloat}
    f = T(factor)
    isfinite(f) && f > 0 || throw(ArgumentError("positive reweighting factor is not representable in accumulator precision"))
    f
end

function _check_reweighted_value(old, new)
    isfinite(old) && isfinite(new) &&
        (iszero(new) ? iszero(old) : abs(new) >= floatmin(typeof(new))) ||
        throw(ArgumentError("reweighted statistic is outside the accumulator's normal-or-zero range"))
    nothing
end


"""
    OnlineUvMean{T<:AbstractFloat}

*BAT-internal, not part of stable public API.*

Univariate mean implemented via Kahan-Babuška-Neumaier summation
([A. Neumaier (1974)](https://doi.org/10.1002/zamm.19740540106)).
"""
mutable struct OnlineUvMean{T<:AbstractFloat}
    sum_v::DoubleFloat{T}
    sum_w::DoubleFloat{T}

    OnlineUvMean{T}() where {T<:AbstractFloat} = new{T}(zero(DoubleFloat{T}), zero(DoubleFloat{T}))

    OnlineUvMean{T}(sum_v::Real, sum_w::Real) where {T<:AbstractFloat} = new{T}(sum_v, sum_w)
end


OnlineUvMean() = OnlineUvMean{Float64}()

@inline Base.getindex(omn::OnlineUvMean{T}) where {T<:AbstractFloat} = T(omn.sum_v / omn.sum_w)


function Base.empty!(target::OnlineUvMean{T}) where T
    target.sum_v = zero(DoubleFloat{T})
    target.sum_w = zero(DoubleFloat{T})

    target
end


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


@inline function Base.push!(omn::OnlineUvMean{T}, data::Real, weight::Real = 1) where {T}
    # Workaround for lack of promotion between, e.g., Float32 and DoubleFloat{Float64}
    weight_conv = T(weight)

    omn.sum_v += DoubleFloat{T}(weight_conv * data)
    omn.sum_w += DoubleFloat{T}(weight_conv)
    omn
end


"""
    reweight_relative!(stats, factor::Real)

Scale historical weights in BAT's internal online statistics. Zero empties all
statistics and metadata; one is an identity. Positive factors scale first
moments and centered scatter by `factor`, and squared weights by `factor^2`,
preserving the chosen weight class's bias correction.

Requires a finite nonnegative factor and gradual underflow. Positive factors
are converted to each accumulator's precision. Unrepresentable conversion or
scaling throws before any component is changed. Nonzero scaled moments,
weight sums and scatter must remain in the normal range; compensated low parts may
round away provided the represented first moment does not vanish. Callers must
have exclusive ownership of `stats`.
"""
function reweight_relative!(target::OnlineUvMean{T}, reweighting_factor::Real) where T
    _reweight_relative!(target, reweighting_factor)
end

function _check_reweight(target::OnlineUvMean{T}, factor) where T
    f = _reweight_factor(T, factor)
    _check_reweighted_value(target.sum_w, target.sum_w * f)
    _check_reweighted_value(target.sum_v, target.sum_v * f)
end

function _apply_reweight!(target::OnlineUvMean{T}, factor) where T
    f = T(factor)
    target.sum_w *= f
    target.sum_v *= f
    target
end



"""
    OnlineUvVar{T<:AbstractFloat,W}

*BAT-internal, not part of stable public API.*

Implementation based on the variance calculation algorithms of
[B. P. Welford (1962)](https://doi.org/10.1080/00401706.1962.10490022)
and [D. H. D. West (1979)](https://doi.org/10.1145/359146.359153).

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


function Base.empty!(target::OnlineUvVar{T,W}) where {T,W}
    target.n = zero(Int64)
    target.sum_w = zero(DoubleFloat{T})
    target.sum_w2 = zero(DoubleFloat{T})
    target.mean_x = zero(T)
    target.s = zero(T)

    target
end


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
        mean_x = (sum_w * DoubleFloat(mean_x) + x.sum_w * DoubleFloat(x.mean_x)) / new_sum_w

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


@inline function Base.push!(ocv::OnlineUvVar{T}, data::Real, weight::Real = 1) where T
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


function reweight_relative!(target::OnlineUvVar{T,W}, reweighting_factor::Real) where {T,W}
    _reweight_relative!(target, reweighting_factor)
end

function _check_reweight(target::OnlineUvVar{T}, factor) where T
    f = _reweight_factor(T, factor)
    _check_reweighted_value(target.sum_w, target.sum_w * f)
    scaled_w2 = target.sum_w2 * f
    _check_reweighted_value(target.sum_w2, scaled_w2)
    _check_reweighted_value(scaled_w2, scaled_w2 * f)
    _check_reweighted_value(target.s, target.s * f)
end

function _apply_reweight!(target::OnlineUvVar{T}, factor) where T
    f = T(factor)
    target.sum_w *= f
    target.sum_w2 = (target.sum_w2 * f) * f
    target.s *= f
    target
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


function Base.empty!(stats::BasicUvStatistics{T,W}) where {T,W}
    empty!(stats.mean)
    empty!(stats.var)
    stats.maximum = typemin(T)
    stats.minimum = typemax(T)

    stats
end


@inline function Base.push!(stats::BasicUvStatistics{T,W}, data::Real, weight::Real = 1) where {T,W}
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


function reweight_relative!(stats::BasicUvStatistics{T,W}, reweighting_factor::Real) where {T,W}
    _reweight_relative!(stats, reweighting_factor)
end

function _check_reweight(stats::BasicUvStatistics, factor)
    _check_reweight(stats.mean, factor)
    _check_reweight(stats.var, factor)
end

function _apply_reweight!(stats::BasicUvStatistics, factor)
    _apply_reweight!(stats.mean, factor)
    _apply_reweight!(stats.var, factor)
    stats
end



const OnlineUvStatistic = Union{BAT.OnlineUvMean, BAT.OnlineUvVar, BAT.BasicUvStatistics}


@inline function Base.append!(stats::OnlineUvStatistic, data::AbstractVector{<:Real})
    push!.(Ref(stats), data)
    stats
end

@inline function Base.append!(stats::OnlineUvStatistic, data::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
    push!.(Ref(stats), data, weights)
    stats
end


Base.merge(target::S, others::S...) where{S <: OnlineUvStatistic} = merge!(deepcopy(target), others...)
