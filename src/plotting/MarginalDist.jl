struct MarginalDist{N,D<:Distribution,VS<:AbstractValueShape}
    dims::NTuple{N,Int}
    dist::D
    origvalshape::VS
end

# TODO: Remove
# function _get_edges(data::Tuple, nbins::Tuple{Vararg{<:Integer}}, closed::Symbol)
#     return StatsBase.histrange(data, StatsBase._nbins_tuple(data, nbins), closed)
# end
#
# function _get_edges(data::Any, nbins::Integer, closed::Symbol)
#     return StatsBase.histrange((data, ), StatsBase._nbins_tuple((data, ), (nbins,)), closed)[1]
# end
#
# function _get_edges(data::Any, nbins::Union{AbstractRange, Tuple{AbstractRange}}, closed::Symbol)
#     return nbins
# end


function bat_marginalize(
    maybe_shaped_samples::DensitySampleVector,
    key::Union{Integer, Symbol, Expr};
    bins::Union{Integer, AbstractRange, AbstractBinning} = FDBinning(),
    closed::Symbol = :left,
    filter::Bool = false,
    normalize = true
)
    samples = BAT.unshaped.(maybe_shaped_samples)

    if filter
        samples = BAT.drop_low_weight_samples(samples)
    end

    idx = asindex(maybe_shaped_samples, key)
    s = flatview(samples.v)[idx, :]

    edges = _bin_edges(samples, idx, bins; closed=closed)

    hist = fit(Histogram,
            s,
            FrequencyWeights(samples.weight),
            edges,
            closed = closed)

    normalize ? hist = StatsBase.normalize(hist) : nothing

    uvbd = EmpiricalDistributions.UvBinnedDist(hist)
    marg = MarginalDist((idx,), uvbd, varshape(maybe_shaped_samples))

    return (result = marg, )
end


function bat_marginalize(
    maybe_shaped_samples::DensitySampleVector,
    key::Union{NTuple{n,Integer}, NTuple{n,Union{Symbol, Expr}}} where n;
    bins = FDBinning(),
    closed::Symbol = :left,
    filter::Bool = false,
    normalize = true
)
    samples = unshaped.(maybe_shaped_samples)

    if filter
        samples = BAT.drop_low_weight_samples(samples)
    end

    idxs = asindex.(Ref(maybe_shaped_samples), key)
    s = Tuple(BAT.flatview(samples.v)[i, :] for i in idxs)

    bins_tuple = isa(bins, Tuple) ? bins : (bins, bins)
    edges = Tuple(_bin_edges(samples, idxs[i], bins_tuple[i], closed=closed) for i in 1:length(bins_tuple))

    hist = fit(Histogram,
            s,
            FrequencyWeights(samples.weight),
            edges,
            closed = closed)

    normalize ? hist = StatsBase.normalize(hist) : nothing

    mvbd = EmpiricalDistributions.MvBinnedDist(hist)
    marg =  MarginalDist(idxs, mvbd, varshape(maybe_shaped_samples))

    return (result = marg, )
end


#for prior
function bat_marginalize(
    prior::NamedTupleDist,
    key::Union{Integer, Symbol};
    bins = FDBinning(),
    edges = nothing,
    closed::Symbol = :left,
    nsamples::Integer = 10^6,
    normalize = true
)
    idx = asindex(prior, key)
    r = rand(prior, nsamples)

    edges = _bin_edges(r[idx, :], bins, closed=closed)

    hist = fit(Histogram, r[idx, :], edges, closed = closed)

    normalize ? hist = StatsBase.normalize(hist) : nothing

    uvbd = EmpiricalDistributions.UvBinnedDist(hist)
    marg = MarginalDist((idx,), uvbd, varshape(prior))

    return (result = marg, )
end


function bat_marginalize(
    prior::NamedTupleDist,
    key::Union{NTuple{2, Symbol}, NTuple{2, Integer}};
    bins = FDBinning(),
    closed::Symbol = :left,
    nsamples::Integer = 10^6,
    normalize=true
)
    idxs = asindex.(Ref(prior), key)

    r = rand(prior, nsamples)
    s = Tuple(r[i, :] for i in idxs)

    bins_tuple = isa(bins, Tuple) ? bins : (bins, bins)
    edges = Tuple(_bin_edges(s[i], bins_tuple[i], closed=closed) for i in 1:length(bins_tuple))

    hist = fit(Histogram,
            s,
            edges,
            closed = closed)

    normalize ? hist = StatsBase.normalize(hist) : nothing

    mvbd = EmpiricalDistributions.MvBinnedDist(hist)
    marg =  MarginalDist(idxs, mvbd, varshape(prior))

    return (result = marg, )
end



function bat_marginalize(
    original::MarginalDist,
    parsel::NTuple{n, Int} where n;
    normalize=true
)
    original_hist = original.dist.h
    dims = collect(1:ndims(original_hist.weights))
    parsel = Tuple(findfirst(x-> x == p, original.dims) for p in parsel)

    weights = sum(original_hist.weights, dims=setdiff(dims, parsel))
    weights = dropdims(weights, dims=Tuple(setdiff(dims, parsel)))

    edges = Tuple([original_hist.edges[p] for p in parsel])
    hist = StatsBase.Histogram(edges, weights, original_hist.closed)

    normalize ? hist = StatsBase.normalize(hist) : nothing

    bd = if length(parsel) == 1
        EmpiricalDistributions.UvBinnedDist(hist)
    else
        EmpiricalDistributions.MvBinnedDist(hist)
    end

    marg = MarginalDist(parsel, bd, original.origvalshape)

    return (result = marg, )
end
