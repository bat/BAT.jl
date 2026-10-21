# This file is a part of BAT.jl, licensed under the MIT License (MIT).

_plot_moment_error(message) = throw(ArgumentError("Binned coordinate moments: $message"))

# Certify the dependency's arithmetic, including intermediates which a later
# density/volume factor could rescue. This deliberately rejects subnormal
# coordinate results; normal floating-point roundoff/cancellation still applies.
function _plot_moment_normal(x::AbstractFloat)
    isfinite(x) && (iszero(x) || abs(x) >= floatmin(typeof(x))) ||
        _plot_moment_error("result is outside the normal-or-zero numerical range")
    return x
end

function _plot_moment_add(a, b)
    result = _plot_moment_normal(a + b)
    iszero(result) && a != -b && _plot_moment_error("addition loses a nonzero result")
    return result
end

function _plot_moment_sub(a, b)
    result = _plot_moment_normal(a - b)
    iszero(result) && a != b && _plot_moment_error("subtraction loses a nonzero result")
    return result
end

function _plot_moment_mul(a, b)
    result = _plot_moment_normal(a * b)
    iszero(result) && !iszero(a) && !iszero(b) &&
        _plot_moment_error("multiplication loses a nonzero result")
    return result
end

function _plot_moment_div(a, b)
    result = _plot_moment_normal(a / b)
    iszero(result) && !iszero(a) && _plot_moment_error("division loses a nonzero result")
    return result
end

function _plot_moment_convert(::Type{T}, value) where {T<:AbstractFloat}
    result = _plot_moment_normal(T(value))
    iszero(result) && !iszero(value) &&
        _plot_moment_error("storage conversion loses a nonzero result")
    return result
end

function _plot_binned_histogram(hist::Histogram)
    get_zero_subnormals() && _plot_moment_error("requires gradual underflow")
    T = float(promote_type(map(eltype, hist.edges)...))
    T <: AbstractFloat || _plot_moment_error("requires floating-point coordinates")
    (T === BigFloat || eltype(hist.weights) === BigFloat) &&
        rounding(BigFloat) != RoundNearest && _plot_moment_error("requires round-to-nearest")
    # Use the constructor's cached-coordinate type for owned edge arrays before
    # normalizing. This also avoids range broadcast/step arithmetic and integer
    # midpoint wrap (the MV constructor otherwise retains the original edges).
    edges = map(edge -> collect(T, edge), hist.edges)
    all(edge -> all(isfinite, edge) && all(>(0), diff(edge)), edges) ||
        _plot_moment_error("stored edges must be finite and strictly increasing")
    return Histogram(edges, copy(hist.weights), hist.closed, hist.isdensity)
end

function _plot_binned_moments(hist::Histogram{<:Real,N}, volumes, univariate::Bool) where N
    Y = hist.weights
    X = map(hist.edges) do edge
        [_plot_moment_div(_plot_moment_add(edge[i], edge[i + 1]), 2)
            for i in 1:length(edge)-1]
    end
    indices = CartesianIndices(Y)
    # Reuse the validated volume per bin: recomputing its N factors inside
    # each covariance entry would add another factor of N to preflight work.
    centers(index) = map((x, i) -> x[i], X, Tuple(index))
    weighted(value, index) = _plot_moment_mul(
        _plot_moment_mul(value, Y[index]), volumes[index])
    add(a, b) = map(_plot_moment_add, a, b)

    # Keep EmpiricalDistributions' expression/reduction order. The existing mass
    # admission handles this denominator, independently of coordinate products.
    total = sum(index -> Y[index] * volumes[index], indices)
    isfinite(total) && total > 0 || _plot_moment_error("invalid normalization total")
    mean_est = map(x -> _plot_moment_div(x, total), mapreduce(
        index -> map(x -> weighted(x, index), centers(index)), add, indices))
    delta(index) = map(_plot_moment_sub, centers(index), mean_est)
    var_est = map(x -> _plot_moment_div(x, total), mapreduce(
        index -> map(x -> weighted(_plot_moment_mul(x, x), index), delta(index)), add, indices))
    cov_est = univariate ? nothing : [
        _plot_moment_div(mapreduce(index ->
            weighted(_plot_moment_mul(
                _plot_moment_sub(X[i][index[i]], mean_est[i]),
                _plot_moment_sub(X[j][index[j]], mean_est[j])), index),
            _plot_moment_add, indices), total) for i in 1:N, j in 1:N
    ]
    mode_index = argmax(Y)
    mode_est = centers(mode_index isa Integer ? CartesianIndex(mode_index) : mode_index)
    T = eltype(first(hist.edges))
    stored(x) = _plot_moment_convert(T, x)
    return (mean=map(stored, mean_est), mode=map(stored, mode_est),
        var=map(stored, var_est), cov=isnothing(cov_est) ? nothing : map(stored, cov_est))
end

function _plot_check_binned_moments(dist, expected, univariate::Bool)
    values = (mean(dist), mode(dist), var(dist))
    reference = (expected.mean, expected.mode, expected.var)
    valid = all(zip(values, reference)) do (actual, checked)
        univariate ? actual == only(checked) :
            size(actual) == (length(checked),) && all(actual .== checked)
    end
    valid || _plot_moment_error("dependency statistics disagree with checked arithmetic")
    all(>=(0), expected.var) || _plot_moment_error("negative variance")
    if !univariate
        covariance = cov(dist)
        covariance == expected.cov && issymmetric(covariance) &&
            all(i -> covariance[i, i] == expected.var[i], eachindex(expected.var)) ||
            _plot_moment_error("dependency covariance disagrees with checked arithmetic")
    end
    return nothing
end
