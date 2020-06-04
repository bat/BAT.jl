struct MarginalDist{N,D<:Distribution,VS<:AbstractValueShape}
    dims::NTuple{N,Int}
    dist::D
    origvalshape::VS
end


#TODO: does not work for unshaped samples
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

    return MarginalDist((idx,), uvbd, varshape(maybe_shaped_samples))
end


function bat_marginalize(
    maybe_shaped_samples::DensitySampleVector,
    key::Union{NTuple{2,Integer}, NTuple{2,Union{Symbol, Expr}}};
    nbins = 200,
    closed::Symbol = :left,
    filter::Bool = false,
    normalize = true
)
    samples = unshaped.(maybe_shaped_samples)

    if filter
        samples = BAT.drop_low_weight_samples(samples)
    end

    i = asindex(maybe_shaped_samples, key[1])
    j = asindex(maybe_shaped_samples, key[2])

    hist = fit(Histogram,
            (flatview(samples.v)[i, :],
            flatview(samples.v)[j, :]),
            FrequencyWeights(samples.weight),
            nbins = nbins, closed = closed)

    normalize ? hist = StatsBase.normalize(hist) : nothing

    mvbd = EmpiricalDistributions.MvBinnedDist(hist)

    return MarginalDist((i,j), mvbd, varshape(maybe_shaped_samples))
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

    return MarginalDist((idx,), uvbd, varshape(prior))
end


function BATHistogram(
    prior::NamedTupleDist,
    params::Union{NTuple{2, Symbol}, NTuple{2, Integer}};
    nbins = 200,
    closed::Symbol = :left,
    nsamples::Integer = 10^6
)
    i = asindex(prior, params[1])
    j = asindex(prior, params[2])

    r = rand(prior, nsamples)
    hist = fit(Histogram, (r[i, :], r[j, :]), nbins = nbins, closed = closed)

    mvbd = EmpiricalDistributions.MvBinnedDist(hist)

    return MarginalDist((i,j), mvbd, varshape(prior))
end
