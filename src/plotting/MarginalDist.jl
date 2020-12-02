struct MarginalDist{N,D<:Distribution,VS<:AbstractValueShape}
    dims::NTuple{N,Int}
    dist::D
    origvalshape::VS
end

function _get_edges(data::Tuple, nbins::Tuple{Vararg{<:Integer}}, closed::Symbol)
    return StatsBase.histrange(data, StatsBase._nbins_tuple(data, nbins), closed)
end

function _get_edges(data::Any, nbins::Integer, closed::Symbol)
    return StatsBase.histrange((data, ), StatsBase._nbins_tuple((data, ), (nbins,)), closed)[1]
end

function _get_edges(data::Any, nbins::Union{AbstractRange, Tuple{AbstractRange}}, closed::Symbol)
    return nbins
end


function get_marginal_dist(
    maybe_shaped_samples::DensitySampleVector,
    key::Union{Integer, Symbol, Expr};
    bins = 200,
    closed::Symbol = :left,
    filter::Bool = false
)
    samples = BAT.unshaped.(maybe_shaped_samples)

    if filter
        samples = BAT.drop_low_weight_samples(samples)
    end

    idx = asindex(maybe_shaped_samples, key)
    s = flatview(samples.v)[idx, :]

    edges = _get_edges(s, bins, closed)

    hist = fit(Histogram,
            s,
            FrequencyWeights(samples.weight),
            edges,
            closed = closed)


    uvbd = EmpiricalDistributions.UvBinnedDist(hist)
    marg = MarginalDist((idx,), uvbd, varshape(maybe_shaped_samples))

    return (result = marg, )
end


function get_marginal_dist(
    maybe_shaped_samples::DensitySampleVector,
    key::Union{NTuple{n,Integer}, NTuple{n,Union{Symbol, Expr}}} where n;
    bins = 200,
    closed::Symbol = :left,
    filter::Bool = false
)
    samples = unshaped.(maybe_shaped_samples)

    if filter
        samples = BAT.drop_low_weight_samples(samples)
    end

    idxs = asindex.(Ref(maybe_shaped_samples), key)
    s = Tuple(BAT.flatview(samples.v)[i, :] for i in idxs)

    edges = if isa(bins, Integer)
        _get_edges(s, (bins,), closed)
    else
        Tuple(_get_edges(s[i], bins[i], closed) for i in 1:length(bins))
    end

    hist = fit(Histogram,
            s,
            FrequencyWeights(samples.weight),
            edges,
            closed = closed)

    mvbd = EmpiricalDistributions.MvBinnedDist(hist)
    marg =  MarginalDist(idxs, mvbd, varshape(maybe_shaped_samples))

    return (result = marg, )
end


#for prior
function get_marginal_dist(
    prior::NamedTupleDist,
    key::Union{Integer, Symbol};
    bins = 200,
    edges = nothing,
    closed::Symbol = :left,
    nsamples::Integer = 10^6
)
    idx = asindex(prior, key)
    r = flatview(unshaped.(rand(prior, nsamples)))

    edges = _get_edges(r[idx, :], bins, closed)

    hist = fit(Histogram, r[idx, :], edges, closed = closed)

    uvbd = EmpiricalDistributions.UvBinnedDist(hist)
    marg = MarginalDist((idx,), uvbd, varshape(prior))

    return (result = marg, )
end


function get_marginal_dist(
    prior::NamedTupleDist,
    key::Union{NTuple{2, Symbol}, NTuple{2, Integer}};
    bins = 200,
    closed::Symbol = :left,
    nsamples::Integer = 10^6
)
    idxs = asindex.(Ref(prior), key)

    r = flatview(unshaped.(rand(prior, nsamples)))
    s = Tuple(r[i, :] for i in idxs)

    edges = if isa(bins, Integer)
        _get_edges(s, (bins,), closed)
    else
        Tuple(_get_edges(s[i], bins[i], closed) for i in 1:length(bins))
    end

    hist = fit(Histogram,
            s,
            edges,
            closed = closed)

    mvbd = EmpiricalDistributions.MvBinnedDist(hist)
    marg =  MarginalDist(idxs, mvbd, varshape(prior))

    return (result = marg, )
end



function get_marginal_dist(
    original::MarginalDist,
    vsel::NTuple{n, Int} where n
)
    original_hist = convert(Histogram, original.dist)
    dims = collect(1:ndims(original_hist.weights))
    vsel = Tuple(findfirst(x-> x == p, original.dims) for p in vsel)

    weights = sum(original_hist.weights, dims=setdiff(dims, vsel))
    weights = dropdims(weights, dims=Tuple(setdiff(dims, vsel)))

    edges = Tuple([original_hist.edges[p] for p in vsel])
    hist = StatsBase.Histogram(edges, weights, original_hist.closed)

    bd = if length(vsel) == 1
        EmpiricalDistributions.UvBinnedDist(hist)
    else
        EmpiricalDistributions.MvBinnedDist(hist)
    end

    marg = MarginalDist(vsel, bd, original.origvalshape)

    return (result = marg, )
end
