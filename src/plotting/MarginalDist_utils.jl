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
