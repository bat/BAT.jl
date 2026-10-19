# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    find_marginalmodes(marg::MarginalDist)

*BAT-internal, not part of stable public API.*

Find the modes of a MarginalDist.
Returns a vector of the bin-centers of the bin(s) with the heighest weight.
"""
function find_marginalmodes(marg::MarginalDist)
    hist = convert(Histogram, marg.dist isa ReshapedDist ? marg.dist.dist : marg.dist)
    dims = ndims(hist.weights)

    max = maximum(hist.weights)
    maxima_idx = findall(x->x==max, hist.weights)

    bin_centers = get_bin_centers(marg)

    return [[bin_centers[d][maxima_idx[i][d]] for d in 1:dims] for i in 1:length(maxima_idx) ]
end


"""
    get_bin_centers(marg::MarginalDist)

*BAT-internal, not part of stable public API.*

Returns a vector of the bin-centers.
"""
function get_bin_centers(marg::MarginalDist)
    hist = convert(Histogram, marg.dist isa ReshapedDist ? marg.dist.dist : marg.dist)
    edges = hist.edges
    dims = ndims(hist.weights)

    centers = [[edges[d][i]+0.5*(edges[d][i+1]-edges[d][i]) for i in 1:length(edges[d])-1] for d in 1:dims]

    return centers
end


function islower(weights, idx)
    if idx==1 && weights[idx]>0
        return true
    elseif weights[idx]>0 && weights[idx-1]==0 && idx < length(weights)
        return true
    else
        return false
    end
end

function isupper(weights, idx)
    if idx==length(weights) && weights[idx-1]>0
        return true
    elseif weights[idx]==0 && weights[idx-1]>0
        return true
    else
        return false
    end
end


# return the lower and upper edges for clusters in which the bincontent is non-zero for all dimensions of a StatsBase.Histogram
# clusters that are seperated <= atol are combined
function get_interval_edges(h::StatsBase.Histogram; atol::Real = 0)
    weights = h.weights
    len = length(weights)

    lower = [h.edges[1][i] for i in 1:len if islower(weights, i)]
    upper = [h.edges[1][i] for i in 2:len if isupper(weights, i)]

    if atol != 0
        idxs = [i for i in 1:length(upper)-1 if lower[i+1]-upper[i] <= atol]
        deleteat!(upper, idxs)
        deleteat!(lower, idxs.+1)
    end

    return lower, upper
end


# These legacy regions rank and accumulate histogram heights. In particular,
# this is not a bin-volume-aware HDR estimator for unequal-width PDF bins.
# Keep that convention here while making its relative arithmetic range-safe.
function _histogram_region_intervals(intervals)
    all(p -> isfinite(p) && 0 <= p <= 1, intervals) ||
        _plot_weight_error("histogram region probabilities must be finite and in [0, 1]")
    return sort(intervals)
end

# for 1d and 2d histograms
function get_smallest_intervals(
    histogram::StatsBase.Histogram,
    intervals::Array{Float64, 1}
)
    intervals = _histogram_region_intervals(intervals)

    hist = deepcopy(histogram)
    dims = size(hist.weights)
    weights = vec(hist.weights)
    rel_weights = _plot_relative_weights(weights; unit_sum=true)

    # Copy selected heights exactly. A Float64 scratch array would corrupt
    # integer heights above 2^53 before assignment back to the histogram.
    hists_weights = [zeros(eltype(weights), length(weights)) for i in eachindex(intervals)]
    weight_ids = sortperm(weights, rev=true) # starting with highest weight

    for (i, intv) in enumerate(intervals)
        sum_weights = zero(eltype(rel_weights))
        for w in weight_ids
            if sum_weights < intv
                hists_weights[i][w] = weights[w]
                sum_weights += rel_weights[w]
            else
                break
            end
        end
    end

    hists = Array{StatsBase.Histogram}(undef, length(intervals))

    for i in 1:length(intervals)
        hists[i] = deepcopy(hist)
        hists[i].weights = reshape(hists_weights[i], dims)
    end

    realintervals = get_probability_content(hist, hists)

    return reverse(hists), reverse(realintervals)
end


# for 1d histograms
function split_central(
    histogram::StatsBase.Histogram,
    intervals::Array{Float64, 1}
)
    intervals = _histogram_region_intervals(intervals)
    intervals = (1 .-intervals)/2

    hist = deepcopy(histogram)
    hists = Array{StatsBase.Histogram}(undef, length(intervals))

    for i in 1:length(intervals)
        hists[i] = deepcopy(hist)
    end

    weights = vec(hist.weights)
    rel_weights = _plot_relative_weights(weights; unit_sum=true)

    for (i, intv) in enumerate(intervals)
        sum_left = zero(eltype(rel_weights))
        sum_right = zero(eltype(rel_weights))

        for l in 1:length(weights)
           if sum_left + rel_weights[l] < intv
                sum_left = sum_left + rel_weights[l]
                hists[i].weights[l] = 0
           else
                break
           end
        end

        for r in length(weights):-1:1
           if sum_right + rel_weights[r] < intv
                sum_right = sum_right + rel_weights[r]
                hists[i].weights[r] = 0
           else
                break
           end
        end
    end

    realintervals = get_probability_content(hist, hists)

    return reverse(hists), reverse(realintervals)
end


# calculate probability percentage enclosed inside the intervals of hists
function get_probability_content(
    hist::StatsBase.Histogram,
    hists::Array{StatsBase.Histogram, 1}
)
    weights = vec(hist.weights)
    relative = _plot_relative_weights(weights)
    totalweight = sum(relative)
    scale = isempty(weights) ? zero(eltype(weights)) : maximum(weights)
    return map(hists) do selected
        # Every subset uses the parent's scale. Normalizing each subset by its
        # own maximum would erase its probability relative to the parent.
        subset = _plot_relative_weights(vec(selected.weights); scale)
        subtotal = sum(subset)
        iszero(totalweight) && return zero(totalweight)
        content = subtotal / totalweight
        (isfinite(content) && (iszero(subtotal) || content > 0)) ||
            _plot_weight_error("histogram region probability is outside the supported numerical range")
        return content
    end
end


function calculate_levels(
    hist::StatsBase.Histogram,
    intervals::Array{<:Real, 1}
)
    intervals = _histogram_region_intervals(intervals)
    levels = Vector{Real}(undef, length(intervals)+1)

    weights = sort(vec(hist.weights), rev=true)

    weight_ids = sortperm(weights, rev=true);
    relative = _plot_relative_weights(weights; unit_sum=true)
    sum_of_weights = _plot_checked_total(weights)

    sum_w = zero(eltype(relative))

    for w_id in weight_ids
        if !isempty(intervals) && sum_w <= intervals[end]
            i = findfirst(x -> sum_w <= x, intervals)
            # One bin can cross several requested probabilities. Initialize
            # every pending level; later bins overwrite exactly the levels
            # reached by the historical strict/overshoot convention.
            levels[i:end-1] .= weights[w_id]
            sum_w += relative[w_id]
        end
    end

    # Empty/all-zero histograms have no region content, including when no
    # iteration above could initialize a level.
    if iszero(sum_of_weights)
        fill!(levels, zero(sum_of_weights))
    else
        cap = 1.1 * sum_of_weights
        (isfinite(cap) && cap > maximum(weights)) ||
            _plot_weight_error("the histogram contour upper level is outside the supported numerical range")
        levels[end] = cap
    end
    return sort(levels)
end
