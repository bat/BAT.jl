# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    find_localmodes(bathist::BATHistogram)

*BAT-internal, not part of stable public API.*

Find the modes of a BATHistogram.
Returns a vector of the bin-centers of the bin(s) with the heighest weight.
"""
function find_localmodes(bathist::BATHistogram)
    dims = ndims(bathist.h.weights)

    max = maximum(bathist.h.weights)
    maxima_idx = findall(x->x==max, bathist.h.weights)

    bin_centers = get_bin_centers(bathist)

    return [[bin_centers[d][maxima_idx[i][d]] for d in 1:dims] for i in 1:length(maxima_idx) ]
end


"""
    get_bin_centers(bathist::BATHistogram)

*BAT-internal, not part of stable public API.*

Returns a vector of the bin-centers.
"""
function get_bin_centers(bathist::BATHistogram)
    edges = bathist.h.edges
    dims = ndims(bathist.h.weights)

    centers = [[edges[d][i]+0.5*(edges[d][i+1]-edges[d][i]) for i in 1:length(edges[d])-1] for d in 1:dims]

    return centers
end


# create a BATHistogram containing some dimensions of higher-dimensional BATHistogram
function subhistogram(
    bathist::BATHistogram,
    params::Array{<:Integer,1}
)
    dims = collect(1:ndims(bathist.h.weights))
    weights = sum(bathist.h.weights, dims=setdiff(dims, params))
    weights = dropdims(weights, dims=Tuple(setdiff(dims, params)))

    edges = Tuple([bathist.h.edges[p] for p in params])
    hist = StatsBase.Histogram(edges, weights, bathist.h.closed)

    return BATHistogram(hist)
end


# return the lower and upper edges for clusters in which the bincontent is non-zero for all dimensions of a BATHistogram
# clusters that are seperated <= atol are combined
function get_cluster_edges(bathist::BATHistogram; atol::Real = 0)
    weights = bathist.h.weights
    len = length(weights)

    lower = [hist.h.edges[1][i] for i in 2:len if (weights[i]>0 && weights[i-1]==0)]
    upper = [hist.h.edges[1][i] for i in 2:len if (weights[i]==0 && weights[i-1]>0)]

    if atol != 0
        idxs = [i for i in 1:length(upper)-1 if lower[i+1]-upper[i] <= atol]
        deleteat!(upper, idxs)
        deleteat!(lower, idxs.+1)
    end

    return lower, upper
end
