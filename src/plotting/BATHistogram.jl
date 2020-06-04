export BATHistogram

mutable struct BATHistogram
    h::StatsBase.Histogram
end


#
# # construct 1D BATHistogram from sample vector
# function BATHistogram(
#     maybe_shaped_samples::DensitySampleVector,
#     key::Union{Integer, Symbol};
#     nbins = 200,
#     closed::Symbol = :left,
#     filter::Bool = false
# )
#     samples = BAT.unshaped.(maybe_shaped_samples)
#
#     if filter
#         samples = BAT.drop_low_weight_samples(samples)
#     end
#
#     idx = asindex(maybe_shaped_samples, key)
#
#     hist = fit(Histogram,
#             flatview(samples.v)[idx, :],
#             FrequencyWeights(samples.weight),
#             nbins = nbins, closed = closed)
#
#     return BATHistogram(hist)
# end



# construct 1D BATHistogram from prior
function BATHistogram(
    prior::NamedTupleDist,
    key::Union{Integer, Symbol};
    nbins = 200,
    closed::Symbol = :left,
    nsamples::Integer = 10^6
)
    idx = asindex(prior, key)
    r = rand(prior, nsamples)
    hist = fit(Histogram, r[idx, :], nbins = nbins, closed = closed)

    return BATHistogram(hist)
end



# construct 2D BATHistogram from sample vector
# function BATHistogram(
#     maybe_shaped_samples::DensitySampleVector,
#     params::Union{NTuple{2, Symbol}, NTuple{2, Integer}};
#     nbins = 200,
#     closed::Symbol = :left,
#     filter::Bool = false
# )
#     samples = unshaped.(maybe_shaped_samples)
#
#     if filter
#         samples = BAT.drop_low_weight_samples(samples)
#     end
#
#     i = asindex(maybe_shaped_samples, params[1])
#     j = asindex(maybe_shaped_samples, params[2])
#
#     hist = fit(Histogram,
#             (flatview(samples.v)[i, :],
#             flatview(samples.v)[j, :]),
#             FrequencyWeights(samples.weight),
#             nbins = nbins, closed = closed)
#
#     return BATHistogram(hist)
# end



# # construct 2D BATHistogram from prior
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

    return BATHistogram(hist)
end
