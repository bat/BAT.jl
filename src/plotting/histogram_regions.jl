# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Histogram regions keep three quantities separate: stored heights, density
# ordering, and integrated cell masses. These helpers do not change the source.
function _histogram_region_requests(intervals::AbstractVector{<:Real})
    all(p -> isfinite(p) && 0 <= p <= 1, intervals) ||
        _plot_weight_error("histogram region probabilities must be finite and in [0, 1]")
    order = sortperm(intervals; rev=true, alg=Base.Sort.MergeSort)
    return collect(intervals[order]), order
end

# Positive factored numbers compare lexicographically by (exponent, significand).
# The zero sentinel sorts below every positive value without ldexp underflow.
function _histogram_region_key(value::T, widths=(); inverse::Bool=false) where {T<:AbstractFloat}
    iszero(value) && return (typemin(Int), zero(T))
    m, e = frexp(value)
    # Deterministic factor ordering also covers a transposition of ND cells.
    for width in sort(collect(widths))
        a, b = frexp(width)
        m = inverse ? m / a : m * a
        e = inverse ? Base.Checked.checked_sub(e, b) : Base.Checked.checked_add(e, b)
        m, adjustment = frexp(m)
        e = Base.Checked.checked_add(e, adjustment)
    end
    return (e, m)
end

# Conservative outward arithmetic for an error bound, not for changing ties or
# selection thresholds. Relative operations use gamma(k); ldexp's final rounding
# adds one smallest positive value per cell, including nonzero subnormals.
_region_up(x::AbstractFloat) = nextfloat(x)
_region_down(x::AbstractFloat) = prevfloat(x)
function _histogram_region_error(::Type{T}, n::Int, dims::Int) where {T<:AbstractFloat}
    n == 0 && return zero(T)
    u, eta = eps(one(T)) / 2, nextfloat(zero(T))
    function gamma(k)
        k == 0 && return zero(T)
        ku = _region_up(_region_up(T(k)) * u)
        denominator = _region_down(one(T) - ku)
        denominator > 0 || _plot_weight_error("histogram region error bound exceeds the working precision")
        return _region_up(ku / denominator)
    end
    # One weight conversion, D exact-endpoint width subtractions, D products.
    rho, g = gamma(Base.Checked.checked_add(Base.Checked.checked_mul(2, dims), 1)), gamma(n - 1)
    dr, dg = _region_down(one(T) - rho), _region_down(one(T) - g)
    dr > 0 && dg > 0 || _plot_weight_error("histogram region error bound exceeds the working precision")
    delta = _region_up(_region_up(_region_up(_region_up(rho / dr) + g) / dg) +
        _region_up(_region_up(_region_up(T(n)) * eta) / dr))
    denominator = _region_down(one(T) - delta)
    denominator > 0 || _plot_weight_error("histogram region error bound exceeds the working precision")
    error = _region_up(_region_up(_region_up(_region_up(4delta) + _region_up(4u)) + _region_up(4eta)) / denominator)
    isfinite(error) && error < 1 || _plot_weight_error("histogram region error bound is not informative")
    return error
end

# Explicit nonnegative sequential sums make the gamma(N-1) bound independent of
# Julia's reduction implementation. No absolute count total is needed here.
function _histogram_region_sum(values::AbstractVector{T}) where {T}
    total = zero(T)
    for value in values
        total += value
    end
    isfinite(total) || _plot_weight_error("histogram region mass total is outside the working numerical range")
    return total
end

function _histogram_region_data(hist::Histogram)
    # Even BigFloat working arithmetic can retain IEEE source heights whose
    # comparisons lose positive subnormals when the current thread flushes them.
    get_zero_subnormals() &&
        _plot_weight_error("histogram regions require subnormal-preserving arithmetic")
    W = eltype(hist.weights)
    W <: Union{Integer,AbstractFloat} ||
        _plot_weight_error("histogram regions require ordinary integer or floating-point heights")
    all(edges -> eltype(edges) <: Union{Integer,AbstractFloat}, hist.edges) ||
        _plot_weight_error("histogram regions require ordinary integer or floating-point edges")
    _plot_validate_weights(vec(hist.weights))
    T = promote_type(Float64, float(W), map(edges -> float(eltype(edges)), hist.edges)...)
    T === Float64 || T === BigFloat ||
        _plot_weight_error("histogram region arithmetic requires a Float64 or BigFloat working type")
    rounding(T) == RoundNearest ||
        _plot_weight_error("histogram regions require round-to-nearest arithmetic")
    widths = map(enumerate(hist.edges)) do (d, edges)
        length(edges) == size(hist.weights, d) + 1 ||
            throw(DimensionMismatch("histogram edges must match the number of cells"))
        converted = T.(collect(edges))
        all(i -> isfinite(converted[i]) && converted[i] == edges[i], eachindex(converted)) ||
            _plot_weight_error("histogram region endpoints must be exactly representable in the working type")
        w = diff(converted)
        all(v -> isfinite(v) && v > 0, w) ||
            _plot_weight_error("histogram region widths must be finite and positive")
        # MPFR has no gradual underflow. A rounded minimum-normal difference
        # need not satisfy the relative subtraction bound used below.
        T === BigFloat && any(v -> v <= floatmin(T), w) &&
            _plot_weight_error("histogram region widths are too close to the working exponent limit")
        w
    end
    weights = vec(hist.weights)
    masskeys = Vector{Tuple{Int,T}}(undef, length(weights))
    densitykeys = similar(masskeys)
    for (i, index) in enumerate(CartesianIndices(hist.weights))
        value = T(weights[i])
        isfinite(value) && (iszero(weights[i]) || value > 0) ||
            _plot_weight_error("histogram height is outside the working numerical range")
        T === BigFloat && 0 < value <= floatmin(T) &&
            _plot_weight_error("histogram heights are too close to the working exponent limit")
        factors = ntuple(d -> widths[d][index[d]], ndims(hist.weights))
        masskeys[i] = _histogram_region_key(value, hist.isdensity ? factors : ())
        densitykeys[i] = _histogram_region_key(value, hist.isdensity ? () : factors; inverse=!hist.isdensity)
    end
    masses = zeros(T, length(weights))
    if any(>(0), weights)
        exponent = first(maximum(masskeys))
        for i in eachindex(masses)
            e, m = masskeys[i]
            if m > 0
                masses[i] = ldexp(m, Base.Checked.checked_add(Base.Checked.checked_sub(e, exponent), 1))
                isfinite(masses[i]) && masses[i] > 0 ||
                    _plot_weight_error("positive histogram region mass is outside the working numerical range")
            end
        end
    end
    total = _histogram_region_sum(masses)
    error = iszero(total) ? zero(T) : _histogram_region_error(T, length(masses), ndims(hist.weights))
    # Direct comparisons preserve original density-height ties/distinctions,
    # including integer values beyond floating-point precision.
    keys = hist.isdensity ? copy(weights) : densitykeys
    return (masses=masses, total=total, density_keys=keys, density_factors=densitykeys, widths=widths, error=error)
end

function _histogram_region_content(data, mask::AbstractVector{Bool})
    length(mask) == length(data.masses) || throw(DimensionMismatch("region mask must match its parent"))
    iszero(data.total) && return zero(data.total)
    subtotal = zero(data.total)
    full = true
    for i in eachindex(mask, data.masses)
        if mask[i]
            subtotal += data.masses[i]
        elseif data.masses[i] > 0
            full = false
        end
    end
    full && return one(data.total)
    iszero(subtotal) && return zero(data.total)
    content = subtotal / data.total
    isfinite(content) && 0 < content <= 1 + data.error ||
        _plot_weight_error("histogram region probability is outside the working numerical range")
    return min(content, one(content))
end

function _histogram_region_probability(p::Real, ::Type{T}) where {T}
    value = T(p)
    isfinite(value) && (iszero(p) || value > 0) ||
        _plot_weight_error("histogram region probability is outside the working numerical range")
    T === BigFloat && 0 < value <= floatmin(T) &&
        _plot_weight_error("histogram region probability is too close to the working exponent limit")
    return value
end

function _histogram_hdr(hist::Histogram, intervals::AbstractVector{<:Real})
    requests, permutation = _histogram_region_requests(intervals)
    data = _histogram_region_data(hist)
    masses, keys = data.masses, data.density_keys
    T = eltype(masses)
    masks = [falses(length(masses)) for _ in requests]
    cutoffs = zeros(Int, length(requests))
    order = filter(i -> masses[i] > 0, sortperm(keys; rev=true))
    # One cumulative pass through complete plateaus, independent of requests.
    ends = Int[]
    cumulative = T[]
    total = zero(T)
    j = 1
    while j <= length(order)
        key = keys[order[j]]
        while j <= length(order) && keys[order[j]] == key
            total += masses[order[j]]
            j += 1
        end
        push!(ends, j - 1)
        push!(cumulative, total)
    end
    group = 1
    for r in reverse(eachindex(requests))
        p = _histogram_region_probability(requests[r], T)
        (iszero(p) || isempty(order)) && continue
        if requests[r] == 1
            group = length(ends)
        else
            target = p * data.total
            while group < length(ends) && cumulative[group] < target
                group += 1
            end
        end
        last = ends[group]
        for i in view(order, 1:last)
            masks[r][i] = true
        end
        cutoffs[r] = order[last]
    end
    contents = [_histogram_region_content(data, mask) for mask in masks]
    return (data=data, requests=requests, order=permutation, masks=masks, contents=contents, cutoff_indices=cutoffs)
end

function _histogram_region_copies(hist::Histogram, masks)
    result = Histogram[]
    for mask in masks
        selected = deepcopy(hist)
        for i in eachindex(selected.weights)
            mask[i] || (selected.weights[i] = zero(eltype(selected.weights)))
        end
        push!(result, selected)
    end
    return result
end

# Whole-bin highest-density superlevel sets, including the entire cutoff plateau.
# Results retain original heights/types and descend in requested probability.
function get_smallest_intervals(hist::Histogram, intervals::AbstractVector{<:Real})
    result = _histogram_hdr(hist, intervals)
    return _histogram_region_copies(hist, result.masks), result.contents
end

# Conservative whole-bin central regions: equality retains a tail boundary bin.
function split_central(hist::Histogram, intervals::AbstractVector{<:Real})
    ndims(hist.weights) == 1 || throw(ArgumentError("central histogram regions require one dimension"))
    requests, _ = _histogram_region_requests(intervals)
    data = _histogram_region_data(hist)
    masses = data.masses
    masks = [falses(length(masses)) for _ in requests]
    for (r, request) in enumerate(requests)
        p = _histogram_region_probability(request, eltype(masses))
        (iszero(p) || iszero(data.total)) && continue
        masks[r] .= masses .> 0
        request == 1 && continue
        target = ((one(p) - p) / 2) * data.total
        for indices in (eachindex(masses), reverse(eachindex(masses)))
            removed = zero(data.total)
            for i in indices
                next = removed + masses[i]
                next < target || break
                masks[r][i] = false
                removed = next
            end
        end
    end
    return _histogram_region_copies(hist, masks), [_histogram_region_content(data, mask) for mask in masks]
end

function get_probability_content(hist::Histogram, selected::AbstractVector{<:Histogram})
    data = _histogram_region_data(hist)
    return map(selected) do child
        size(child.weights) == size(hist.weights) && child.edges == hist.edges &&
            child.closed == hist.closed && child.isdensity == hist.isdensity ||
            throw(ArgumentError("histogram regions must use the parent's grid and density convention"))
        all(i -> iszero(child.weights[i]) || child.weights[i] == hist.weights[i], eachindex(hist.weights)) ||
            throw(ArgumentError("histogram regions must contain only original parent heights or zero"))
        _histogram_region_content(data, vec(child.weights .> 0))
    end
end

# Bound the additional dense smoothing payload and multiply-add count before
# constructing a kernel or padded grid. These are per-contour resource limits.
const _HISTOGRAM_SMOOTH_MAX_KERNEL_CELLS = 1 << 20
const _HISTOGRAM_SMOOTH_MAX_BUFFER_CELLS = 1 << 23
const _HISTOGRAM_SMOOTH_MAX_WORK = 1 << 26

_histogram_smoothing_sigmas(smoothing::Real) = (smoothing, smoothing)
_histogram_smoothing_sigmas(smoothing::Tuple{Real,Real}) = smoothing

function _histogram_smoothing_plan(dims::Tuple{Int,Int}, sigmas::Tuple{Float64,Float64})
    all(>(0), dims) || throw(ArgumentError("histogram smoothing dimensions must be positive"))
    all(s -> isfinite(s) && s >= 0, sigmas) ||
        throw(ArgumentError("histogram contour smoothing must be finite and nonnegative"))
    kernel_limit = _HISTOGRAM_SMOOTH_MAX_KERNEL_CELLS
    buffer_limit = _HISTOGRAM_SMOOTH_MAX_BUFFER_CELLS
    work_limit = _HISTOGRAM_SMOOTH_MAX_WORK
    fail(kind) = throw(ArgumentError("histogram contour smoothing exceeds the $kind budget; reduce smoothing or the number of bins, or use whole-bin regions"))

    # Each side alone must fit the kernel cell budget. Compare before ceil(Int)
    # or multiplication, including finite sigma outside the machine Int range.
    all(s -> s <= (kernel_limit - 1) ÷ 4, sigmas) || fail("kernel")
    sides = map(s -> 4 * ceil(Int, s) + 1, sigmas)
    # gaussian_kernel squares native-integer offsets before floating division.
    # Wide one-axis kernels can fit the resource budgets but overflow on Int32.
    all(l -> l ÷ 2 <= isqrt(typemax(Int)), sides) ||
        throw(ArgumentError("histogram contour smoothing support exceeds the working integer range; reduce smoothing"))
    sides[1] <= kernel_limit ÷ sides[2] || fail("kernel")
    kernel_cells = sides[1] * sides[2]
    dims[1] <= buffer_limit ÷ dims[2] || fail("buffer")
    cells = dims[1] * dims[2]
    cells <= work_limit ÷ kernel_cells || fail("work")

    all(d -> dims[d] <= buffer_limit - sides[d] + 1, 1:2) || fail("buffer")
    padded = map((n, l) -> n + l - 1, dims, sides)
    padded[1] <= buffer_limit ÷ padded[2] || fail("buffer")
    padded_cells = padded[1] * padded[2]
    # Kernel and its positive-value copy, positive field copy, convolution
    # result, relative normalization, padded input and Gaussian vectors.
    remaining = buffer_limit
    for (multiplicity, count) in ((2, kernel_cells), (3, cells), (1, padded_cells), (2, sides[1]), (2, sides[2]))
        count <= remaining ÷ multiplicity || fail("buffer")
        remaining -= multiplicity * count
    end
    return (sides=sides, kernel_cells=kernel_cells, padded=padded,
        buffer_cells=buffer_limit - remaining, work=cells * kernel_cells)
end

# Paired contour field and scientific metadata. Contour geometry is interpolated,
# not an exact boundary of the selected histogram cells. No backend types here.
function _histogram_contour_data(hist::Histogram, intervals::AbstractVector{<:Real}; smoothing::Union{Real,Tuple{Real,Real}}=0)
    ndims(hist.weights) == 2 || throw(ArgumentError("histogram contours require two dimensions"))
    sigmas = _histogram_smoothing_sigmas(smoothing)
    all(s -> isfinite(s) && s >= 0, sigmas) || throw(ArgumentError("histogram contour smoothing must be finite and nonnegative"))
    requests, permutation = _histogram_region_requests(intervals)
    data = _histogram_region_data(hist)
    field = zeros(Float64, size(hist.weights))
    cutoffs = Union{Nothing,Float64}[nothing for _ in requests]
    # No displayed field is demanded for empty/zero requests or zero mass.
    if iszero(data.total) || !any(>(0), requests)
        return (density=field, requests=requests, contents=zeros(eltype(data.masses), length(requests)), cutoffs=cutoffs, order=permutation)
    end
    all(>=(2), size(field)) || throw(ArgumentError("histogram contours need at least two cells per axis; use whole-bin regions instead"))
    largest_e, largest_m = maximum(data.density_factors)
    for i in eachindex(field)
        e, m = data.density_factors[i]
        if m > 0
            value = ldexp(m / largest_m, Base.Checked.checked_sub(e, largest_e))
            field[i] = _plot_float64(value, "relative contour density")
            field[i] > 0 || _plot_weight_error("positive relative contour density is outside the renderer range")
        end
    end
    if any(s -> !iszero(s), sigmas)
        converted = map(s -> _plot_float64(s, "histogram contour smoothing"), sigmas)
        plan = _histogram_smoothing_plan(size(field), converted)
        kernel = smoothing isa Real ? gaussian_kernel(converted[1]; l=plan.sides[1]) :
            gaussian_kernel(converted; l=plan.sides)
        all(v -> isfinite(v) && v >= 0, kernel) && any(>(0), kernel) ||
            _plot_weight_error("histogram contour smoothing kernel is outside the working numerical range")
        minimum(filter(>(0), vec(field))) * minimum(filter(>(0), vec(kernel))) > 0 ||
            _plot_weight_error("positive contour smoothing products are outside the working numerical range")
        smoothed = convolution(field, kernel; padding=:same)
        all(i -> isfinite(smoothed[i]) && smoothed[i] >= 0 && (iszero(field[i]) || smoothed[i] > 0), eachindex(field)) ||
            _plot_weight_error("contour smoothing loses positive density")
        field = reshape(_plot_relative_weights(vec(smoothed)), size(field))
    end
    displayed = Histogram(deepcopy(hist.edges), field, hist.closed, true)
    result = _histogram_hdr(displayed, intervals)
    for r in eachindex(cutoffs)
        index = result.cutoff_indices[r]
        index == 0 || (cutoffs[r] = field[index])
    end
    return (density=field, requests=result.requests, contents=result.contents, cutoffs=cutoffs, order=result.order)
end
