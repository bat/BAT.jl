# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    find_localmodes(marg::MarginalDist)

*BAT-internal, not part of stable public API.*

Find the modes of a MarginalDist.
Returns a vector of the bin-centers of the bin(s) with the heighest weight.
"""
function find_localmodes(marg::MarginalDist)
    hist = marg.dist.h
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
    hist = marg.dist.h
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


# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# for 1d and 2d histogramsm
function get_smallest_intervals(
    histogram::StatsBase.Histogram,
    intervals::Array{Float64, 1}
)
    intervals = sort(intervals)

    hist = deepcopy(histogram)
    dims = size(hist.weights)
    weights = vec(hist.weights)
    totalweight = sum(weights)
    rel_weights = weights/totalweight

    hists_weights = [zeros(length(weights)) for i in 1:length(intervals)]
    weight_ids = sortperm(weights, rev=true) # starting with highest weight

    for (i, intv) in enumerate(intervals)
        sum_weights = 0.
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
    intervals = sort(intervals)
    intervals = (1 .-intervals)/2

    hist = deepcopy(histogram)
    hists = Array{StatsBase.Histogram}(undef, length(intervals))

    for i in 1:length(intervals)
        hists[i] = deepcopy(hist)
    end

    weights = vec(hist.weights)
    totalweight = sum(weights)
    rel_weights = weights/totalweight

    for (i, intv) in enumerate(intervals)
        sum_left = 0.
        sum_right = 0.

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
    totalweight = sum(hist.weights)
    return [sum(hists[i].weights)/totalweight for i in 1:length(hists)]
end



function calculate_levels(
    hist::StatsBase.Histogram,
    intervals::Array{<:Real, 1}
)
    intervals = sort(intervals)
    levels = Vector{Real}(undef, length(intervals)+1)

    weights = sort(vec(hist.weights), rev=true)

    weight_ids = sortperm(weights, rev=true);
    sum_of_weights = sum(weights)

    sum_w = 0.0

    for w_id in weight_ids
        if(sum_w <= intervals[end])
            i = findfirst(x -> sum_w <= x, intervals)
            levels[i] = weights[w_id]
            sum_w += weights[w_id]/sum_of_weights
        end
    end

    levels[end] = 1.1*sum_of_weights
    return sort(levels)
end
