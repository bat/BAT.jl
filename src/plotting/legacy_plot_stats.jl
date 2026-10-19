# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Legacy Plots overlays use the same online components as MCMCBasicStats.
# Their frequency correction depends on absolute counts: do not normalize them.
# Work only on selected coordinates; unused covariance/log-density calculations
# must not determine whether a mean or a global-mode marker can be drawn.
function _legacy_plot_moments(samples::DensitySampleVector, idxs; spread::Bool=false)
    total = _plot_checked_total(samples.weight)
    total > 0 || throw(ArgumentError("Legacy plotting statistics require positive total weight"))
    !spread || total > 1 || throw(ArgumentError(
        "Legacy frequency-corrected spread requires total weight greater than one"))

    values = flatview(unshaped.(samples).v)
    P = promote_type(eltype(values), eltype(samples.weight), Float64)
    P <: AbstractFloat || _plot_weight_error(
        "Legacy plotting statistics require a supported floating-point accumulator")
    indices = collect(idxs)
    positive = findall(>(0), samples.weight)
    coords = Matrix{P}(undef, length(indices), length(positive))
    weights = Vector{P}(undef, length(positive))
    for (j, row) in enumerate(positive)
        weights[j] = P(samples.weight[row])
        isfinite(weights[j]) && weights[j] > 0 || _plot_weight_error(
            "Legacy statistics cannot represent a positive weight in their accumulator")
        for (i, dim) in enumerate(indices)
            value = values[dim, row]
            isfinite(value) || throw(ArgumentError(
                "Legacy plotting statistics require finite positive-weight coordinates"))
            coords[i, j] = P(value)
            isfinite(coords[i, j]) && (iszero(value) || !iszero(coords[i, j])) ||
                _plot_weight_error("Legacy statistics cannot represent a selected coordinate")
        end
    end
    # Conversion can change the accumulator range and its addition order.
    _plot_checked_total(weights)
    means = OnlineMvMean{P}(length(indices))
    covariance = spread ? OnlineMvCov{P,FrequencyWeights}(length(indices)) : nothing
    for j in axes(coords, 2)
        point = view(coords, :, j)
        push!(means, point, weights[j])
        isnothing(covariance) || push!(covariance, point, weights[j])
    end
    mu = [_plot_float64(means[i], "legacy plotting mean") for i in eachindex(indices)]
    deviation = Float64[]
    if spread
        for (i, dim) in enumerate(indices)
            variance = covariance[i, i]
            isfinite(variance) && variance >= 0 || _plot_weight_error(
                "Legacy frequency-corrected variance is outside the supported numerical range")
            constant = all(row -> values[dim, row] == values[dim, first(positive)], positive)
            iszero(variance) && !constant && _plot_weight_error(
                "Legacy variance underflowed for a nonconstant selected coordinate")
            dev = _plot_float64(sqrt(variance), "legacy plotting standard deviation")
            _legacy_plot_interval(mu[i], dev)
            push!(deviation, dev)
        end
    end
    return (mean=mu, std=deviation)
end

function _legacy_plot_interval(mu::Real, dev::Real)
    lo = _plot_float64(mu - dev, "legacy plotting interval lower endpoint")
    hi = _plot_float64(mu + dev, "legacy plotting interval upper endpoint")
    dev > 0 && !(lo < hi) && _plot_weight_error(
        "Legacy plotting interval collapsed at the selected coordinate scale")
    return lo, hi
end

# Match MCMCBasicStats' strict comparison (first maximum wins), including its
# consideration of zero-weight rows. The score scan itself needs no weight sum.
function _legacy_plot_mode(samples::DensitySampleVector, idxs)
    score = typemin(promote_type(eltype(samples.logd), Float64))
    winner = nothing
    for row in eachindex(samples)
        value = samples.logd[row]
        if value > score
            winner = row
        end
        score = max(score, value)
    end
    isnothing(winner) && throw(ArgumentError(
        "Legacy global-mode overlay has no selected mode (for example, all log densities are -Inf)"))
    values = flatview(unshaped.(samples).v)
    return [_plot_float64(values[dim, winner], "legacy global-mode coordinate") for dim in idxs]
end

function _legacy_plot_line_height(marg::MarginalDist)
    hist = convert(Histogram, marg.dist isa ReshapedDist ? marg.dist.dist : marg.dist)
    peak = maximum(hist.weights)
    return _plot_float64(peak * 1.03, "legacy overlay height (1.03 times peak density)")
end

function _legacy_scatter_coordinates(samples::DensitySampleVector, idxs)
    # Rejected samples have nonpositive weights in this historical scatter
    # encoding and remain visible. Do not impose distribution-weight semantics.
    all(isfinite, samples.weight) || throw(ArgumentError("Legacy scatter weights must be finite"))
    values = flatview(samples.v)
    return [[_plot_float64(values[dim, row], "legacy scatter coordinate")
        for row in eachindex(samples)] for dim in idxs]
end

function _legacy_scatter_color(color, weight)
    weight >= 1 && return color
    alpha = color.alpha * weight
    color.alpha > 0 && weight > 0 && iszero(alpha) && _plot_weight_error(
        "Legacy scatter opacity underflowed for a positive weight")
    return RGBA(convert(RGB, color), _plot_float64(alpha, "legacy scatter opacity"))
end
