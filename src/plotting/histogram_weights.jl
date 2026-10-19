# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function _plot_histogram_mode(mode::Symbol)
    mode in (:none, :density, :probability, :pdf) ||
        throw(ArgumentError("Normalization mode must be :pdf, :density, :probability or :none"))
    return mode
end

function _plot_validate_coordinates(cols::Tuple, weights::AbstractVector)
    all(v -> length(v) == length(weights), cols) || throw(DimensionMismatch("coordinates and weights must have matching lengths"))
    for col in cols, (value, weight) in zip(col, weights)
        if weight > 0
            value isa Real && isfinite(value) || _plot_weight_error("positive-weight coordinates must be finite real numbers")
        end
    end
    return nothing
end

function _plot_histogram_widths(hist::Histogram)
    widths = map(edges -> diff(collect(float.(edges))), hist.edges)
    all(w -> all(v -> isfinite(v) && v > 0, w), widths) ||
        _plot_weight_error("histogram edges must have finite positive widths")
    return widths
end

function _plot_validate_histogram(hist::Histogram)
    _plot_validate_weights(vec(hist.weights))
    _plot_histogram_widths(hist)
    return hist
end

function _plot_fit_histogram(cols::Tuple, weights::AbstractVector, edges::Tuple, mode::Symbol; closed::Symbol=:left)
    _plot_histogram_mode(mode)
    _plot_validate_weights(weights)
    _plot_validate_coordinates(cols, weights)
    prepared, total = if mode in (:pdf, :probability)
        u = _plot_relative_weights(weights)
        (u, sum(u))
    else
        count = _plot_checked_total(weights)
        (typeof(count).(weights), count)
    end
    hist = fit(Histogram, cols, FrequencyWeights(prepared, total), edges; closed=closed)
    _plot_validate_histogram(hist)
    return _plot_normalize_histogram(hist, mode)
end

# Multiply without forming an intermediate bin volume. Used only when an
# integral/mass is itself required; density division uses the quotient helper.
function _plot_positive_product(value::Real, factors::Real...)
    T = promote_type(Float64, float(typeof(value)), map(v -> float(typeof(v)), factors)...)
    isfinite(value) && value >= 0 || _plot_weight_error("histogram value must be finite and nonnegative")
    all(v -> isfinite(v) && v > 0, factors) || _plot_weight_error("histogram widths must be finite and positive")
    iszero(value) && return zero(T)
    m, e = frexp(T(value))
    for v in factors
        a, b = frexp(T(v))
        m *= a
        e = Base.Checked.checked_add(e, b)
        m, adjustment = frexp(m)
        e = Base.Checked.checked_add(e, adjustment)
    end
    result = ldexp(m, e)
    isfinite(result) && result > 0 || _plot_weight_error("histogram mass is outside the working numerical range")
    return result
end

function _plot_normalize_histogram(hist::Histogram, mode::Symbol)
    _plot_histogram_mode(mode)
    _plot_validate_histogram(hist)
    if mode == :none
        hist.isdensity || _plot_checked_total(vec(hist.weights))
        return deepcopy(hist)
    end
    widths = _plot_histogram_widths(hist)
    relative_mode = mode in (:pdf, :probability)
    mode == :density && hist.isdensity && return deepcopy(hist)
    T = promote_type(Float64, float(eltype(hist.weights)))
    values = T.(hist.weights)
    if relative_mode
        # A fit of prepared relative weights already has a safe total. Scaling
        # its bins again can erase a subnormal occupied bin when several unit
        # weights accumulated in another bin. Preserve those heights so the
        # factored PDF quotient can rescue a small mass over a small width.
        representable = all(i -> isfinite(values[i]) &&
            (iszero(hist.weights[i]) || values[i] > 0), eachindex(values))
        if hist.isdensity || !representable || !isfinite(sum(values))
            values = reshape(_plot_relative_weights(vec(hist.weights)), size(hist.weights))
        end
    end
    if !relative_mode
        _plot_checked_total(vec(hist.weights))
    end
    # Zero mass has a useful empty-display interpretation, but does not become
    # a probability distribution. MarginalDist separately requires positive mass.
    total = if relative_mode && hist.isdensity
        masses = map(CartesianIndices(values)) do index
            _plot_positive_product(values[index], map((w, i) -> w[i], widths, Tuple(index))...)
        end
        _plot_checked_total(vec(masses))
    elseif relative_mode
        sum(values)
    else
        one(eltype(values))
    end
    if iszero(total)
        return Histogram(deepcopy(hist.edges), zero(values), hist.closed, hist.isdensity)
    end
    result = map(CartesianIndices(values)) do index
        factors = mode in (:pdf, :density) && !hist.isdensity ? map((w, i) -> w[i], widths, Tuple(index)) : ()
        denominators = relative_mode ? (total, factors...) : factors
        _plot_positive_ratio(values[index], denominators...)
    end
    isdensity = hist.isdensity || mode in (:pdf, :density)
    return Histogram(deepcopy(hist.edges), result, hist.closed, isdensity)
end

# EmpiricalDistributions normalizes again and multiplies volumes in a different
# order from StatsBase. Guard that actual dependency boundary, then verify the
# returned PDF against the prepared bin probabilities. Cached coordinate moments
# have a separate inherited range limitation (H22).
function _plot_binned_dist(probabilities::Histogram, univariate::Bool)
    any(>(0), probabilities.weights) || _plot_weight_error("a probability distribution requires positive in-bin mass")
    widths = _plot_histogram_widths(probabilities)
    for index in CartesianIndices(probabilities.weights)
        volume = prod(map((w, i) -> w[i], widths, Tuple(index)))
        isfinite(volume) && volume > 0 ||
            _plot_weight_error("bin volume is outside the binned distribution's numerical range")
    end
    # Preflight the normalization which the public dependency constructor uses,
    # including zero bins; this prevents uninformative assertions/NaN PDFs.
    normalized = StatsBase.normalize(probabilities; mode=:pdf)
    _plot_validate_histogram(normalized)
    all(i -> iszero(probabilities.weights[i]) || normalized.weights[i] > 0, eachindex(probabilities.weights)) ||
        _plot_weight_error("binned distribution normalization loses positive mass")
    dependency_masses = map(CartesianIndices(normalized.weights)) do index
        normalized.weights[index] * prod(map((w, i) -> w[i], widths, Tuple(index)))
    end
    _plot_validate_weights(vec(dependency_masses))
    all(i -> iszero(probabilities.weights[i]) || dependency_masses[i] > 0, eachindex(dependency_masses)) ||
        _plot_weight_error("binned distribution volume arithmetic loses positive mass")
    sum(dependency_masses) ≈ 1 || _plot_weight_error("binned distribution volume arithmetic has no unit probability mass")
    dist = try
        univariate ? EmpiricalDistributions.UvBinnedDist(probabilities) : EmpiricalDistributions.MvBinnedDist(probabilities)
    catch err
        err isa AssertionError || rethrow()
        _plot_weight_error("binned distribution constructor cannot represent this histogram")
    end
    actual = convert(Histogram, dist)
    _plot_validate_histogram(actual)
    masses = map(CartesianIndices(actual.weights)) do index
        _plot_positive_product(actual.weights[index], map((w, i) -> w[i], widths, Tuple(index))...)
    end
    T = promote_type(Float64, eltype(masses), eltype(probabilities.weights))
    # Edges may have Float64 precision even when the weights are BigFloat.
    tolerance = 64 * eps(Float64) * max(1, length(masses))
    all(i -> isapprox(masses[i], probabilities.weights[i]; rtol=tolerance, atol=zero(T)), eachindex(masses)) ||
        _plot_weight_error("binned distribution normalization does not preserve bin probabilities")
    isapprox(sum(masses), one(T); rtol=tolerance, atol=zero(T)) ||
        _plot_weight_error("binned distribution has no unit probability mass")
    return dist
end
