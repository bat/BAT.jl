# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Pure numerical helpers for the four Makie KDE recipes. Bounds are hard
# support, not a viewport. Outputs own their arrays; inputs are never mutated.
const _KDE_CUTOFF_BANDWIDTHS = 4.0 # omitted Gaussian image tails are small, not zero
const _KDE_FFT_PADDING = 8.0
const _KDE_MAX_POINTS_1D = 65536
const _KDE_MAX_AXIS_2D = 8192
const _KDE_MAX_CELLS_2D = 4194304
# Arithmetic admission, not an accuracy bound for the inherited FFT estimator.
const _KDE_UNBOUNDED_RTOL = 2.0^-20

_kde_numerical_error(message) = throw(ArgumentError("KDE: $message"))

function _kde_support_bounds(support)
    isnothing(support) && return (-Inf, Inf)
    length(support) == 2 || throw(ArgumentError("KDE support requires two bounds"))
    rawlo, rawhi = support
    rawlo isa Real && rawhi isa Real || throw(ArgumentError("KDE support bounds must be real numbers"))
    rawlo < rawhi || throw(ArgumentError("KDE support must have lo < hi and no NaN bounds"))
    lo, hi = Float64(rawlo), Float64(rawhi)
    ((isfinite(rawlo) && !isfinite(lo)) || (isfinite(rawhi) && !isfinite(hi))) &&
        throw(ArgumentError("KDE support bounds must be representable as Float64"))
    lo < hi || throw(ArgumentError("KDE support has no positive Float64 width"))
    return (lo, hi)
end

_support_is_binding(lo::Real, hi::Real) = isfinite(lo) || isfinite(hi)

# Keep legacy arithmetic, but do not lose a positive row solely because division
# took place in Float16/Float32 before the Float64 convolution population filter.
function _rescale_kde_weights(weights::AbstractVector)
    isempty(weights) && return weights
    largest = maximum(weights)
    relative = weights ./ largest
    for (original, scaled) in zip(weights, relative)
        if original > 0 && iszero(scaled) &&
                (original isa Union{Float16,Float32} || largest isa Union{Float16,Float32}) &&
                Float64(original) / Float64(largest) > 0
            _kde_numerical_error(
                "relative weight was lost in source precision; convert weights to Float64 before plotting"
            )
        end
    end
    return relative
end

function _kde_population(coords::Tuple, weights::AbstractVector, bounds::Tuple)
    all(v -> length(v) == length(weights), coords) ||
        throw(ArgumentError("KDE coordinates and weights must have matching lengths"))
    isempty(weights) && throw(ArgumentError("KDE requires positive mass"))
    all(w -> w isa Real && isfinite(w) && w >= 0, weights) ||
        throw(ArgumentError("KDE weights must be finite and nonnegative"))
    any(>(0), weights) || throw(ArgumentError("KDE requires positive mass"))
    # Validate original positive mass before relative weights can underflow.
    for (v, (lo, hi)) in zip(coords, bounds), i in eachindex(weights)
        if weights[i] > 0
            v[i] isa Real && isfinite(v[i]) && lo <= v[i] <= hi ||
                throw(ArgumentError("KDE positive-weight coordinates must be finite and inside hard support"))
            isfinite(Float64(v[i])) ||
                throw(ArgumentError("KDE coordinates must be representable as Float64"))
        end
    end
    w = _rescale_kde_weights(weights)
    keep = map(v -> Float64(v) > 0, w)
    if all(keep)
        return coords, w
    end
    return map(v -> v[keep], coords), w[keep]
end

# Weighted counterpart of KernelDensity.default_bandwidth: uncorrected weighted
# standard deviation/IQR and Kish effective count. Preserve its existing fallback.
function _weighted_kde_bandwidth(values::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, alpha::Float64 = 0.9)
    isempty(weights) && return nothing
    w_max = maximum(weights)
    (isfinite(w_max) && w_max > 0) || return nothing
    w_rel = weights ./ w_max
    sum_w = sum(w_rel)
    sum_w2 = sum(abs2, w_rel)
    (isfinite(sum_w) && isfinite(sum_w2) && sum_w > 0 && sum_w2 > 0) || return nothing
    n_eff = sum_w^2 / sum_w2
    n_eff <= 1 && return nothing
    w = Weights(w_rel)
    var_width = std(values, w)
    q25, q75 = quantile(values, w, [0.25, 0.75])
    width = min(var_width, (q75 - q25) / 1.34)
    if width == 0.0
        width = var_width == 0.0 ? 1.0 : var_width
    end
    isfinite(width) || return nothing
    return alpha * width * n_eff^(-0.2)
end

function _kde_bandwidth(values::AbstractVector, weights::AbstractVector)
    h = allequal(weights) ? nothing : _weighted_kde_bandwidth(values, weights)
    return isnothing(h) ? KernelDensity.default_bandwidth(values) : h
end

# Enclose quartile interpolation and weighted rank arithmetic. Exact original
# plateaus retain zero IQR; unresolved intervals are rejected by the caller.
_kde_iqr_bounds(values, weights, centered, weighted) =
    _kde_iqr_bounds(values, weights, centered, weighted, nothing)

function _kde_iqr_bounds(values, weights, centered, weighted, coordinate_bounds)
    coordinate(i, upper) = isnothing(coordinate_bounds) ? centered[i] : coordinate_bounds[upper ? 2 : 1][i]
    order = weighted ? sortperm(collect(zip(values, weights))) : sortperm(values)
    n = length(order)
    down(x) = prevfloat(x)
    up(x) = nextfloat(x)
    function interpolate(j, fraction, upper)
        j <= 1 && return coordinate(first(order), upper)
        alo, ahi = coordinate(order[j-1], false), coordinate(order[j-1], true)
        blo, bhi = coordinate(order[j], false), coordinate(order[j], true)
        a = upper ? ahi : alo
        values[order[j-1]] == values[order[j]] && return a
        isnothing(coordinate_bounds) && !(alo < bhi) && return upper ? Inf : -Inf
        round = upper ? up : down
        gap = max(0.0, upper ? up(bhi-alo) : down(blo-ahi))
        return clamp(round(a + max(0.0, round(fraction*gap))), alo, bhi)
    end
    if !weighted
        offset, remainder = divrem(n-1, 4)
        positions = ((1+offset, remainder/4),
            (n-offset-(remainder>0), remainder == 0 ? 0.0 : 1-remainder/4))
        quartiles = map(positions) do (i, fraction)
            fraction == 0 && return (coordinate(order[i], false), coordinate(order[i], true), i, i)
            return (interpolate(i+1, fraction, false), interpolate(i+1, fraction, true), i, i+1)
        end
    else
        lower = Float64.(weights)
        upper = copy(lower)
        for i in eachindex(weights)
            lower[i] > weights[i] && (lower[i] = max(0.0, down(lower[i])))
            upper[i] < weights[i] && (upper[i] = up(upper[i]))
        end
        total_lo = total_hi = 0.0
        for i in order
            total_lo = max(0.0, down(total_lo + lower[i]))
            total_hi = up(total_hi + upper[i])
        end
        isfinite(total_hi) || return (-Inf, Inf)
        quartiles = map((0.25, 0.75)) do p
            target_lo = max(0.0, down(max(0.0, down(p*total_lo)) +
                max(0.0, down((1-p)*lower[first(order)]))))
            target_hi = up(up(p*total_hi) + up((1-p)*upper[first(order)]))
            prefix_lo = prefix_hi = 0.0
            qlo, qhi = coordinate(first(order), false), coordinate(last(order), true)
            left, right = 1, n
            found_lo = found_hi = false
            for j in eachindex(order)
                i = order[j]
                next_lo = max(0.0, down(prefix_lo + lower[i]))
                next_hi = up(prefix_hi + upper[i])
                if !found_lo && next_hi > target_lo
                    fraction = clamp(down(down(target_lo-prefix_hi)/upper[i]), 0.0, 1.0)
                    qlo = interpolate(j, fraction, false)
                    left = max(1, j-1)
                    found_lo = true
                end
                if !found_hi && next_lo > target_hi
                    fraction = iszero(lower[i]) ? 1.0 :
                        clamp(up(up(target_hi-prefix_lo)/lower[i]), 0.0, 1.0)
                    qhi = interpolate(j, fraction, true)
                    right = j
                    found_hi = true
                end
                prefix_lo, prefix_hi = next_lo, next_hi
                found_lo && found_hi && break
            end
            return (qlo, qhi, left, right)
        end
    end
    q25, q75 = quartiles
    all(isfinite, (q25[1], q25[2], q75[1], q75[2])) || return (-Inf, Inf)
    if !isnothing(coordinate_bounds) && values[order[q25[3]]] == values[order[q75[4]]]
        return (0.0, 0.0)
    end
    if q25[1] == q25[2] == q75[1] == q75[2]
        return values[order[q25[3]]] == values[order[q75[4]]] ? (0.0, 0.0) : (-Inf, Inf)
    end
    return (max(0.0, down(q75[1]-q25[2])), up(q75[2]-q25[1]))
end

# Preserve the selected bandwidth; witness arithmetic only decides admission.
function _kde_bounded_bandwidth(values, weights)
    fail(message) = _kde_numerical_error(
        "bounded estimate $message; convert coordinates and weights to Float64, recenter or change units, or use a histogram")
    weighted = !allequal(weights)
    selected = weighted ? _weighted_kde_bandwidth(values, weights) : KernelDensity.default_bandwidth(values)
    isnothing(selected) && fail("cannot resolve its effective sample count or bandwidth")
    h = Float64(selected)
    isfinite(h) && h > 0 || fail("has an invalid bandwidth")
    length(values) <= 1 && return h

    centered = collect(Float64, values)
    relative_weights = weighted ? weights ./ maximum(weights) : weights
    reference_weights = Float64.(relative_weights)
    constant = all(v -> v == first(values), values)
    # Center near the quartiles before scaling, preserving small local gaps.
    origin = constant ? first(centered) : weighted ?
        only(quantile(centered, Weights(reference_weights), [0.25])) : quantile(centered, 0.25)
    centered .-= origin
    span = constant ? 1.0 : maximum(abs, centered)
    isfinite(span) && span > 0 || fail("cannot resolve its centered span")
    for i in eachindex(centered)
        scaled = centered[i] / span
        !iszero(centered[i]) && iszero(scaled) && fail("loses a centered deviation in reference scaling")
        centered[i] = scaled
    end
    count = weighted ? sum(reference_weights)^2 / sum(abs2, reference_weights) : length(values)
    isfinite(count) && count > 1 || fail("cannot resolve its reference effective count")
    widths = (1.0, 1.0)
    if !constant
        sd = weighted ? std(centered, Weights(reference_weights)) : std(centered)
        _kde_positive_normal(sd * sd) || fail("cannot resolve its reference variance")
        iqr_lo, iqr_hi = _kde_iqr_bounds(values, relative_weights, centered, weighted)
        if iszero(iqr_lo) && iszero(iqr_hi)
            widths = (sd, sd)
        else
            0 < iqr_lo <= iqr_hi < Inf || fail("cannot resolve its interquartile width")
            widths = (min(sd, iqr_lo/1.34), min(sd, iqr_hi/1.34))
        end
    end
    for width in widths
        reference = 0.9 * width * count^(-0.2)
        isfinite(reference) && reference > 0 &&
            abs((h / span) / reference - 1) <= 2.0^-20 || fail("cannot resolve its bandwidth reliably")
    end
    return h
end

function _kde_images(z, lo, hi)
    images = (z,)
    z - lo < _KDE_CUTOFF_BANDWIDTHS && (images = (images..., lo - (z - lo)))
    hi - z < _KDE_CUTOFF_BANDWIDTHS && (images = (images..., hi + (hi - z)))
    return images
end

function _kde_grid_size(span, points_per_bandwidth, minimum_points, maximum_points)
    required = points_per_bandwidth * span + 1
    (isfinite(required) && 1 < required <= maximum_points) ||
        _kde_numerical_error(
            "required grid size $required exceeds the $maximum_points point limit; use a histogram for this population"
        )
    n = nextpow(2, max(minimum_points, ceil(Int, required)))
    n <= maximum_points || _kde_numerical_error("required grid size $n exceeds the $maximum_points point limit")
    return n
end

# Scan image extrema before allocating augmented samples or an FFT grid. Finite
# remote bounds may map to infinities; they need not be convolution endpoints.
function _kde_axis_plan(values, weights, bounds, points_per_bandwidth, minimum_points, maximum_points)
    h = _kde_bounded_bandwidth(values, weights)
    origin = Float64(first(values))
    z = (Float64.(values) .- origin) ./ h
    all(isfinite, z) || _kde_numerical_error("data span cannot be represented in bandwidth units")
    lo, hi = (bounds .- origin) ./ h
    a, b = Inf, -Inf
    for v in z, image in _kde_images(v, lo, hi)
        a = min(a, image)
        b = max(b, image)
    end
    a -= _KDE_FFT_PADDING
    b += _KDE_FFT_PADDING
    n = _kde_grid_size(b - a, points_per_bandwidth, minimum_points, maximum_points)
    grid = range(a, b; length = n)
    all(i -> grid[i] < grid[i+1], 1:n-1) || _kde_numerical_error("FFT grid is not representable")
    left, right = max(a, lo), min(b, hi)
    left < right || _kde_numerical_error("support interval is not representable in bandwidth units")
    return (; origin, h, z, lo, hi, bounds, grid, left, right, points_per_bandwidth)
end

function _kde_physical_grid(plan, z)
    x = [muladd(plan.h, v, plan.origin) for v in z]
    first(z) == plan.lo && (x[1] = plan.bounds[1])
    last(z) == plan.hi && (x[end] = plan.bounds[2])
    all(isfinite, x) || _kde_numerical_error("physical grid has nonfinite coordinates")
    widths = diff(x)
    all(w -> isfinite(w) && w > 0, widths) ||
        _kde_numerical_error("physical grid has indistinguishable or invalid coordinates")
    maximum(widths) / plan.h <= (1 + 8eps(Float64)) / plan.points_per_bandwidth ||
        _kde_numerical_error("physical grid cannot resolve the bandwidth")
    return x
end

# Linear interpolation is nonnegative and has explicit integration semantics.
# Endpoint clamping here only accounts for mapping a grid endpoint to physical
# Float64 and back; admission never clamps observations to support.
function _kde_lerp(grid, values, z)
    j = clamp(searchsortedlast(grid, z), 1, length(grid)-1)
    t = clamp((z - grid[j]) / (grid[j+1] - grid[j]), 0.0, 1.0)
    return (1-t) * values[j] + t * values[j+1]
end

function _kde_relative_widths(x)
    widths = diff(x)
    all(w -> isfinite(w) && w > 0, widths) || _kde_numerical_error("cell widths must be finite and positive")
    scale = maximum(widths)
    relative = widths ./ scale
    all(>(0), relative) || _kde_numerical_error("relative cell widths are not representable")
    return relative, scale
end

function _kde_relative_density(density)
    all(v -> isfinite(v) && v >= 0, density) || _kde_numerical_error("density must be finite and nonnegative")
    peak = maximum(density)
    peak > 0 || _kde_numerical_error("density has no positive mass")
    return density ./ peak, peak
end

# Factor a product without overflowing/underflowing it. Numerators below are
# relative heights in [0,1]; applying the final exponent avoids unsafe h_x*h_y.
function _kde_scale_product(scales...)
    mantissa, exponent = 1.0, 0
    for scale in scales
        isfinite(scale) && scale > 0 || _kde_numerical_error("normalization has no finite positive mass")
        m, e = frexp(scale)
        mantissa *= m
        exponent += e
    end
    m, e = frexp(mantissa)
    return m, exponent + e
end

function _kde_normalized_density(relative, scales...)
    m, e = _kde_scale_product(scales...)
    result = map(v -> ldexp(v / m, -e), relative)
    all(isfinite, result) && maximum(result) > 0 ||
        _kde_numerical_error("normalized density is not representable as Float64")
    return result
end

function _kde_normalize_curve(x, density)
    relative, _ = _kde_relative_density(density)
    widths, scale = _kde_relative_widths(x)
    mass = sum(widths[i] * (relative[i]/2 + relative[i+1]/2) for i in eachindex(widths))
    return _kde_normalized_density(relative, mass, scale)
end

function _kde_cell_mass(density, xedges, yedges)
    relative, _ = _kde_relative_density(density)
    wx, sx = _kde_relative_widths(xedges)
    wy, sy = _kde_relative_widths(yedges)
    mass = [relative[i,j] * wx[i] * wy[j] for i in eachindex(wx), j in eachindex(wy)]
    total = sum(mass)
    isfinite(total) && total > 0 || _kde_numerical_error("cell probabilities have no finite positive mass")
    return (; relative, probabilities = mass ./ total, total, sx, sy)
end

_kde_unbounded_error(message) = _kde_numerical_error(
    "unbounded estimate $message; convert coordinates and weights to Float64, recenter or change units, or use a histogram")

function _kde_positive_normal(value)
    v = float(value)
    return isfinite(v) && v >= floatmin(typeof(v))
end

function _kde_check_unbounded_bandwidth(values, weights, h, weighted)
    isfinite(h) && h > 0 || _kde_unbounded_error("has an invalid bandwidth")
    length(values) <= 1 && return nothing
    narrowed_h = Float64(h)
    isfinite(narrowed_h) && narrowed_h > 0 || _kde_unbounded_error("cannot represent its bandwidth witness")
    witness_h = (narrowed_h > h ? prevfloat(narrowed_h) : narrowed_h,
        narrowed_h < h ? nextfloat(narrowed_h) : narrowed_h)
    sd = weighted ? std(values, Weights(weights ./ maximum(weights))) : std(values)
    isfinite(sd) || _kde_unbounded_error("has nonfinite variance")
    relative_weights = weighted ? weights ./ maximum(weights) : weights
    reference_weights = weighted ? Float64.(relative_weights) : nothing
    count = weighted ? sum(reference_weights)^2 / sum(abs2, reference_weights) : length(values)
    isfinite(count) && count > 1 || _kde_unbounded_error("cannot resolve its reference effective count")
    if allequal(values)
        iszero(sd) || _kde_unbounded_error("cannot resolve constant-data variance")
        reference = 0.9 * count^(-0.2)
        all(v -> abs(v / reference - 1) <= _KDE_UNBOUNDED_RTOL, witness_h) ||
            _kde_unbounded_error("cannot resolve its constant-data bandwidth reliably")
        return nothing
    end
    _kde_positive_normal(sd * sd) || _kde_unbounded_error("cannot resolve its variance")

    # Witness only: preserve the original selected bandwidth and KDE call. A
    # finite normal variance can still be wrong after low-precision mean rounding.
    centered = collect(Float64, values)
    centered .-= first(centered)
    span = maximum(abs, centered)
    isfinite(span) && span > 0 || _kde_unbounded_error("cannot resolve its centered span")
    centered ./= span
    reference = weighted ? _weighted_kde_bandwidth(centered, reference_weights) :
        KernelDensity.default_bandwidth(centered)
    !isnothing(reference) && isfinite(reference) && reference > 0 &&
        abs((h / span) / reference - 1) <= _KDE_UNBOUNDED_RTOL ||
        _kde_unbounded_error("cannot resolve its bandwidth reliably")
    # Start again from the source: the first-origin witness can lose local
    # quartile gaps. Both enclosure endpoints must support the selected h.
    centered .= values
    origin = weighted ? only(quantile(centered, Weights(reference_weights), [0.25])) :
        quantile(centered, 0.25)
    centered .-= origin
    span = maximum(abs, centered)
    isfinite(span) && span > 0 || _kde_unbounded_error("cannot resolve its quartile-centered span")
    lower, upper = similar(centered), similar(centered)
    coordinate_error = 0.0
    for i in eachindex(centered)
        narrowed = Float64(values[i])
        lo = narrowed > values[i] ? prevfloat(narrowed) : narrowed
        hi = narrowed < values[i] ? nextfloat(narrowed) : narrowed
        lower[i] = prevfloat(prevfloat(lo-origin) / span)
        upper[i] = nextfloat(nextfloat(hi-origin) / span)
        scaled = centered[i] / span
        !iszero(centered[i]) && iszero(scaled) &&
            _kde_unbounded_error("loses a centered deviation in reference scaling")
        centered[i] = scaled
        coordinate_error = max(coordinate_error,
            nextfloat(scaled-lower[i]), nextfloat(upper[i]-scaled))
    end
    reference_sd = weighted ? std(centered, Weights(reference_weights)) : std(centered)
    _kde_positive_normal(reference_sd * reference_sd) ||
        _kde_unbounded_error("cannot resolve its reference variance")
    # SD changes by at most E in the weighted population norm, or sqrt(2)E
    # for corrected SD. Use 2E to cover coordinate conversion/conditioning.
    sd_error = nextfloat(2 * coordinate_error)
    sd_lo = max(0.0, prevfloat(reference_sd-sd_error))
    sd_hi = nextfloat(reference_sd+sd_error)
    iqr_lo, iqr_hi = _kde_iqr_bounds(values, relative_weights, centered, weighted, (lower, upper))
    if iszero(iqr_lo) && iszero(iqr_hi)
        widths = (sd_lo, sd_hi)
    else
        0 < iqr_lo <= iqr_hi < Inf || _kde_unbounded_error("cannot resolve its interquartile width")
        widths = (min(sd_lo, iqr_lo/1.34), min(sd_hi, iqr_hi/1.34))
    end
    for width in widths
        reference = 0.9 * width * count^(-0.2)
        isfinite(reference) && reference > 0 &&
            all(v -> abs((v / span) / reference - 1) <= _KDE_UNBOUNDED_RTOL, witness_h) ||
            _kde_unbounded_error("cannot resolve its bandwidth reliably")
    end
    return nothing
end

function _kde_check_unbounded_axis(values, grid, h)
    # Range algorithms may inspect nominal spacing instead of rounded nodes.
    physical = collect(Float64, grid)
    s = step(grid)
    length(physical) >= 2 && all(isfinite, physical) && _kde_positive_normal(s) ||
        _kde_unbounded_error("has an invalid physical grid")
    all(d -> isfinite(d) && d > 0 && abs(d/s - 1) <= _KDE_UNBOUNDED_RTOL, diff(physical)) ||
        _kde_unbounded_error("cannot represent uniform physical grid spacing")

    # KernelDensity uses Normal(0.0,h), whose CF separately squares sigma and t.
    sigma = promote(0.0, h)[2]
    frequency = -2pi / (s * length(grid))
    last_frequency = (length(grid) ÷ 2) * frequency
    all(_kde_positive_normal, (sigma*sigma/2, frequency*frequency, last_frequency*last_frequency)) ||
        _kde_unbounded_error("exceeds Gaussian convolution arithmetic range")
    for value in values
        k = searchsortedfirst(grid, value)
        j = k - 1
        1 <= j < length(grid) || _kde_unbounded_error("cannot locate a positive-weight observation on its grid")
        left, right = (grid[k]-value)/s, (value-grid[j])/s
        isfinite(left) && isfinite(right) && left >= 0 && right >= 0 &&
            abs(left + right - 1) <= _KDE_UNBOUNDED_RTOL ||
            _kde_unbounded_error("cannot resolve its observation bins")
    end
    return s
end

function _kde_check_unbounded_tabulation(weights, steps::Tuple)
    total = sum(weights)
    _kde_positive_normal(total) || _kde_unbounded_error("has an invalid weight total")
    area = prod(steps)
    if length(steps) == 1
        first_product = total * only(steps)
        denominator = first_product * only(steps)
        factors = (first_product, denominator)
    else
        squared_area = area * area
        denominator = total * squared_area
        factors = (area, squared_area, denominator)
    end
    all(_kde_positive_normal, factors) && _kde_positive_normal(1.0 / denominator) ||
        _kde_unbounded_error("exceeds tabulation arithmetic range")
    # Deposits and FFT buffers are Float64 even for BigFloat input arithmetic.
    # A subnormal total can quantize relative masses while still summing to one.
    fft_total = Float64(inv(area))
    _kde_positive_normal(fft_total) && isfinite(16 * fft_total) ||
        _kde_unbounded_error("exceeds the Float64 FFT buffer range")
    return nothing
end

function _kde_check_unbounded_density(density, steps::Tuple)
    relative, peak = _kde_relative_density(density)
    # This is an invariant check, never a normalization of the returned density.
    m, e = _kde_scale_product(peak, sum(relative), steps...)
    mass = ldexp(m, e)
    isfinite(mass) && abs(mass - 1) <= 8 * _KDE_UNBOUNDED_RTOL ||
        _kde_unbounded_error("does not preserve probability mass")
    return nothing
end

function _kde_check_unbounded_edges(edges, s)
    physical = collect(Float64, edges)
    all(isfinite, physical) && all(d -> isfinite(d) && d > 0 &&
        abs(d/s - 1) <= 2 * _KDE_UNBOUNDED_RTOL, diff(physical)) ||
        _kde_unbounded_error("cannot represent its physical cell edges")
    return nothing
end

function _weighted_kde1d(values::AbstractVector, weights::AbstractVector, support = nothing)
    bounds = _kde_support_bounds(support)
    (values,), weights = _kde_population((values,), weights, (bounds,))
    if !_support_is_binding(bounds...)
        weighted = !allequal(weights)
        h = weighted ? _weighted_kde_bandwidth(values, weights) : nothing
        weighted && isnothing(h) && _kde_unbounded_error("cannot resolve its effective sample count or bandwidth")
        selected_h = isnothing(h) ? KernelDensity.default_bandwidth(values) : h
        _kde_check_unbounded_bandwidth(values, weights, selected_h, weighted)
        k = isnothing(h) ? kde(values; weights) : kde(values; weights, bandwidth = h)
        steps = (_kde_check_unbounded_axis(values, k.x, selected_h),)
        _kde_check_unbounded_tabulation(weights, steps)
        _kde_check_unbounded_density(k.density, steps)
        return (x = k.x, density = k.density)
    end
    p = _kde_axis_plan(values, weights, bounds, 16, 2048, _KDE_MAX_POINTS_1D)
    aug, aw = Float64[], Float64[]
    for i in eachindex(p.z), v in _kde_images(p.z[i], p.lo, p.hi)
        push!(aug, v)
        push!(aw, weights[i])
    end
    k = kde(aug, p.grid; weights = aw, bandwidth = 1.0)
    inside = p.grid[searchsortedlast(p.grid, p.left)+1:searchsortedfirst(p.grid, p.right)-1]
    z = vcat(p.left, collect(inside), p.right)
    x = _kde_physical_grid(p, z)
    density = [_kde_lerp(k.x, k.density, (v-p.origin)/p.h) for v in x]
    return (x = x, density = _kde_normalize_curve(x, density))
end

function _kde_output_cells(p)
    n = max(1, ceil(Int, (p.right - p.left) / step(p.grid)))
    n <= length(p.grid) || _kde_numerical_error("output cells exceed the FFT grid budget")
    edges = _kde_physical_grid(p, range(p.left, p.right; length = n+1))
    # Evaluate conceptual centers in local units even if a physical midpoint of
    # adjacent floats rounds to an endpoint. Explicit edges still render a cell.
    zedges = (edges .- p.origin) ./ p.h
    centers = [zedges[i]/2 + zedges[i+1]/2 for i in 1:n]
    x = [edges[i] + (edges[i+1]-edges[i])/2 for i in 1:n]
    return (; edges, centers, x)
end

function _kde_center_edges(x)
    edges = vcat(first(x) - (x[2]-x[1])/2,
        [x[i] + (x[i+1]-x[i])/2 for i in 1:length(x)-1],
        last(x) + (x[end]-x[end-1])/2)
    _kde_relative_widths(edges)
    return edges
end

function _weighted_kde2d(x::AbstractVector, y::AbstractVector, weights::AbstractVector, support = nothing)
    isnothing(support) || length(support) == 2 || throw(ArgumentError("2D KDE requires support for two coordinates"))
    bounds = isnothing(support) ? ((-Inf,Inf), (-Inf,Inf)) :
        (_kde_support_bounds(support[1]), _kde_support_bounds(support[2]))
    (x,y), weights = _kde_population((x,y), weights, bounds)
    if !any(b -> _support_is_binding(b...), bounds)
        hx = hy = nothing
        weighted = !allequal(weights)
        if weighted
            hx = _weighted_kde_bandwidth(x, weights)
            hy = _weighted_kde_bandwidth(y, weights)
        end
        use_default = isnothing(hx) || isnothing(hy)
        weighted && use_default && _kde_unbounded_error("cannot resolve its effective sample count or bandwidth")
        selected_hx, selected_hy = use_default ? KernelDensity.default_bandwidth((x,y)) : (hx,hy)
        _kde_check_unbounded_bandwidth(x, weights, selected_hx, weighted)
        _kde_check_unbounded_bandwidth(y, weights, selected_hy, weighted)
        k = use_default ? kde((x,y); weights) :
            kde((x,y); weights, bandwidth = (hx,hy))
        steps = (_kde_check_unbounded_axis(x, k.x, selected_hx), _kde_check_unbounded_axis(y, k.y, selected_hy))
        _kde_check_unbounded_tabulation(weights, steps)
        _kde_check_unbounded_density(k.density, steps)
        xedges, yedges = _kde_center_edges(k.x), _kde_center_edges(k.y)
        _kde_check_unbounded_edges(xedges, steps[1])
        _kde_check_unbounded_edges(yedges, steps[2])
        return (x = k.x, y = k.y, density = k.density, xedges = xedges, yedges = yedges)
    end
    px = _kde_axis_plan(x, weights, bounds[1], 8, 256, _KDE_MAX_AXIS_2D)
    py = _kde_axis_plan(y, weights, bounds[2], 8, 256, _KDE_MAX_AXIS_2D)
    length(px.grid) <= _KDE_MAX_CELLS_2D ÷ length(py.grid) ||
        _kde_numerical_error(
            "required 2D grid $(length(px.grid)) × $(length(py.grid)) exceeds the $_KDE_MAX_CELLS_2D cell limit; use a histogram for this population"
        )
    ax, ay, aw = Float64[], Float64[], Float64[]
    for i in eachindex(weights), vx in _kde_images(px.z[i], px.lo, px.hi), vy in _kde_images(py.z[i], py.lo, py.hi)
        push!(ax, vx)
        push!(ay, vy)
        push!(aw, weights[i])
    end
    k = kde((ax,ay), (px.grid,py.grid); weights = aw, bandwidth = (1.0,1.0))
    cx, cy = _kde_output_cells(px), _kde_output_cells(py)
    # Separable interpolation work is bounded by the same tensor grid budget.
    along_x = [_kde_lerp(k.x, view(k.density,:,j), z) for z in cx.centers, j in eachindex(k.y)]
    density = [_kde_lerp(k.y, view(along_x,i,:), z) for i in eachindex(cx.centers), z in cy.centers]
    masses = _kde_cell_mass(density, cx.edges, cy.edges)
    normalized = _kde_normalized_density(masses.relative, masses.total, masses.sx, masses.sy)
    return (x = cx.x, y = cy.x, density = normalized, xedges = cx.edges, yedges = cy.edges)
end

# Superlevel mass of a continuous linear curve, in relative height/width units.
function _kde_superlevel_mass(widths, y, threshold)
    total = 0.0
    for i in eachindex(widths)
        a, b = y[i], y[i+1]
        if min(a,b) >= threshold
            total += widths[i] * (a/2 + b/2)
        elseif max(a,b) > threshold
            high, low = max(a,b), min(a,b)
            fraction = (high-threshold)/(high-low)
            total += (widths[i]*fraction) * (high/2 + threshold/2)
        end
    end
    return total
end

function _kde_curve_region(x, density, p)
    0 < p <= 1 || throw(ArgumentError("KDE region content must be in (0,1]"))
    length(x) == length(density) && length(x) >= 2 ||
        throw(ArgumentError("KDE curve needs matching coordinates and heights"))
    y, peak = _kde_relative_density(density)
    widths, scale = _kde_relative_widths(x)
    total = _kde_superlevel_mass(widths, y, 0.0)
    isfinite(total) && total > 0 || _kde_numerical_error("curve has no representable positive mass")
    target = p * total
    threshold = 1.0
    if _kde_superlevel_mass(widths, y, threshold) < target
        lo, hi = 0.0, 1.0
        for _ in 1:64
            mid = lo/2 + hi/2
            (mid == lo || mid == hi) && break
            if _kde_superlevel_mass(widths, y, mid) >= target
                lo = mid
            else
                hi = mid
            end
        end
        threshold = lo
    end
    components = NamedTuple{(:x,:y),Tuple{Vector{Float64},Vector{Float64}}}[]
    for i in 1:length(x)-1
        a, b = y[i], y[i+1]
        max(a,b) < threshold && continue
        left, right = Float64(x[i]), Float64(x[i+1])
        yl, yr = Float64(density[i]), Float64(density[i+1])
        if min(a,b) < threshold
            fraction = (threshold-a)/(b-a)
            crossing = muladd(fraction, right-left, left)
            # Round crossings outward. Returned geometry contains the requested
            # continuous region; achieved mass below uses the actual vertices.
            represented = (crossing-left)/(right-left)
            if a < threshold
                represented > fraction && (crossing = max(left, prevfloat(crossing)))
                f = (crossing-left)/(right-left)
                left, yl = crossing, (1-f)*density[i] + f*density[i+1]
            else
                represented < fraction && (crossing = min(right, nextfloat(crossing)))
                f = (crossing-left)/(right-left)
                right, yr = crossing, (1-f)*density[i] + f*density[i+1]
            end
        end
        left < right || continue
        if !isempty(components) && last(components[end].x) == left
            push!(components[end].x, right)
            push!(components[end].y, yr)
        else
            push!(components, (x = [left,right], y = [yl,yr]))
        end
    end
    area = sum(sum((c.x[i+1]-c.x[i])/scale * (c.y[i]/peak/2 + c.y[i+1]/peak/2)
        for i in 1:length(c.x)-1) for c in components; init = 0.0)
    return (threshold = threshold*peak, achieved = area/total, components = components)
end
