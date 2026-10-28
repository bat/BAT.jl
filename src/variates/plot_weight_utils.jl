# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Local arithmetic for plotting/reporting. Stored sample weights, sampler
# statistics and the core _canonical_rel_weights convention are unchanged.
_plot_weight_error(message) = throw(ArgumentError("Plot weights: $message"))

function _plot_validate_weights(w::AbstractVector{<:Real})
    all(v -> isfinite(v) && v >= zero(v), w) ||
        _plot_weight_error("weights must be finite and nonnegative")
    return nothing
end

function _plot_relative_weights(w::AbstractVector{<:Real}; unit_sum::Bool=false, scale=nothing)
    _plot_validate_weights(w)
    T = isnothing(scale) ? promote_type(Float64, float(eltype(w))) :
        promote_type(Float64, float(eltype(w)), float(typeof(scale)))
    isempty(w) && return T[]
    largest = maximum(w)
    iszero(largest) && return zeros(T, length(w))
    divisor = isnothing(scale) ? largest : scale
    isfinite(divisor) && divisor >= largest ||
        _plot_weight_error("relative scale must be finite and at least the largest weight")
    # In particular, widen Float16/32 before division; logarithmic weights must
    # instead normalize before conversion, as in the existing BAT helper.
    u = if eltype(w) <: Rational
        # Fixed-width rational products can overflow before conversion.
        T.(w) ./ T(divisor)
    else
        _normalize_rel_weights(w, divisor, T)
    end
    all(i -> isfinite(u[i]) && (iszero(w[i]) || u[i] > 0), eachindex(w, u)) ||
        _plot_weight_error("positive relative mass is outside the working numerical range")
    if unit_sum
        total = sum(u)
        isfinite(total) && total > 0 || _plot_weight_error("relative total is outside the working numerical range")
        u ./= total
        all(i -> iszero(w[i]) || u[i] > 0, eachindex(w, u)) ||
            _plot_weight_error("positive probability is outside the working numerical range")
    end
    return u
end

_plot_count_type(::Type{T}) where {T<:Integer} = typeof(sum(T[]))
_plot_count_type(::Type{T}) where {T<:Real} = promote_type(Float64, float(T))

function _plot_checked_total(w::AbstractVector{<:Real})
    _plot_validate_weights(w)
    T = _plot_count_type(eltype(w))
    total = zero(T)
    for value in w
        v = T(value)
        isfinite(v) && (iszero(value) || v > 0) ||
            _plot_weight_error("absolute weight is outside the count accumulator's numerical range")
        total = T <: Integer && T !== BigInt ? Base.Checked.checked_add(total, v) : total + v
        isfinite(total) || throw(OverflowError("Plot weights: absolute count total is not representable; use probability or PDF normalization"))
    end
    return total
end

# Preserve source ordering and cutoff ties, even when division rounds distinct
# original values to the same relative weight. A zero cutoff retains zeros.
function _plot_low_weight_mask(w::AbstractVector{<:Real}, fraction::Real=1e-5, threshold::Real=1e-2)
    isfinite(fraction) && 0 <= fraction <= 1 || _plot_weight_error("filter fraction must be between zero and one")
    isfinite(threshold) && threshold >= 0 || _plot_weight_error("filter threshold must be finite and nonnegative")
    u = _plot_relative_weights(w)
    (isempty(u) || all(iszero, u)) && return trues(length(w))
    minimum(u) / maximum(u) > threshold && return trues(length(w))
    order = sortperm(w)
    cumulative = cumsum(u[order])
    # The prepared total is at least one. Compare in those units instead of
    # dividing tiny cumulative values to zero (which could drop positive rows
    # even when fraction == 0).
    i = searchsortedlast(cumulative, fraction * last(cumulative))
    return iszero(i) ? trues(length(w)) : w .>= w[order[i]]
end

function _plot_float64(value::Real, label::AbstractString)
    result = Float64(value)
    isfinite(result) && (iszero(value) || !iszero(result)) ||
        _plot_weight_error("$label is not representable as finite Float64 without losing a nonzero value")
    return result
end

# A quotient whose denominator product need not be representable. Normalize the
# mantissa after every division, then check only the required final result.
function _plot_positive_ratio(value::Real, denominators::Real...)
    T = promote_type(Float64, float(typeof(value)), map(d -> float(typeof(d)), denominators)...)
    isfinite(value) && value >= 0 || _plot_weight_error("histogram mass must be finite and nonnegative")
    all(d -> isfinite(d) && d > 0, denominators) ||
        _plot_weight_error("histogram widths and normalization totals must be finite and positive")
    iszero(value) && return zero(T)
    m, e = frexp(T(value))
    for d in denominators
        a, b = frexp(T(d))
        m /= a
        e = Base.Checked.checked_sub(e, b)
        m, adjustment = frexp(m)
        e = Base.Checked.checked_add(e, adjustment)
    end
    result = ldexp(m, e)
    isfinite(result) && result > 0 || _plot_weight_error("histogram value is outside the working numerical range")
    return result
end
