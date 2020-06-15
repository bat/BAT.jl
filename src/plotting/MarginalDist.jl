struct MarginalDist{N,D<:Distribution,VS<:AbstractValueShape}
    dims::NTuple{N,Int}
    dist::D
    origvalshape::VS
end


function bat_marginalize(
    maybe_shaped_samples::DensitySampleVector,
    key::Union{Integer, Symbol, Expr};
    nbins = 200,
    closed::Symbol = :left,
    filter::Bool = false,
    normalize = true
)
    samples = BAT.unshaped.(maybe_shaped_samples)

    if filter
        samples = BAT.drop_low_weight_samples(samples)
    end

    idx = asindex(maybe_shaped_samples, key)

    hist = fit(Histogram,
            flatview(samples.v)[idx, :],
            FrequencyWeights(samples.weight),
            nbins = nbins, closed = closed)

    normalize ? hist = StatsBase.normalize(hist) : nothing

    uvbd = EmpiricalDistributions.UvBinnedDist(hist)
    marg = MarginalDist((idx,), uvbd, varshape(maybe_shaped_samples))

    return (result = marg, )
end


function bat_marginalize(
    maybe_shaped_samples::DensitySampleVector,
    key::Union{NTuple{n,Integer}, NTuple{n,Union{Symbol, Expr}}} where n;
    nbins = 200,
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

    hist = fit(Histogram,
            s,
            FrequencyWeights(samples.weight),
            nbins = nbins, closed = closed)

    normalize ? hist = StatsBase.normalize(hist) : nothing

    mvbd = EmpiricalDistributions.MvBinnedDist(hist)
    marg =  MarginalDist(idxs, mvbd, varshape(maybe_shaped_samples))

    return (result = marg, )
end


#for prior
function bat_marginalize(
    prior::NamedTupleDist,
    key::Union{Integer, Symbol};
    nbins = 200,
    closed::Symbol = :left,
    nsamples::Integer = 10^6,
    normalize = true
)
    idx = asindex(prior, key)
    r = rand(prior, nsamples)

    hist = fit(Histogram, r[idx, :], nbins = nbins, closed = closed)
    normalize ? hist = StatsBase.normalize(hist) : nothing

    uvbd = EmpiricalDistributions.UvBinnedDist(hist)
    marg = MarginalDist((idx,), uvbd, varshape(prior))

    return (result = marg, )
end


function bat_marginalize(
    prior::NamedTupleDist,
    key::Union{NTuple{2, Symbol}, NTuple{2, Integer}};
    nbins = 200,
    closed::Symbol = :left,
    nsamples::Integer = 10^6,
    normalize=true
)
    idxs = asindex.(Ref(prior), key)

    r = rand(prior, nsamples)
    s = Tuple(r[i, :] for i in idxs)

    hist = fit(Histogram, s, nbins = nbins, closed = closed)

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
