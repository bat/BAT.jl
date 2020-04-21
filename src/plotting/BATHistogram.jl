export BATHistogram

mutable struct BATHistogram
    h::StatsBase.Histogram
end

function get_histogram(bathistogram::BATHistogram)
    return bathistogram.h
end


# get index of parameter name in shaped samples
function get_param_index(samples::DensitySampleVector, param::Symbol)
    if isa(varshape(samples), NamedTupleShape)
        i = findfirst(x -> x == param, keys(samples[1].v))
        return i
    else
        throw(ArgumentError("Samples are unshaped. Key :$param cannot be matched. Use index instead."))
    end
end

# get index of parameter name in prior
function get_param_index(prior::NamedTupleDist, param::Symbol)
    i = findfirst(x -> x == param, keys(prior))
end

# get index of parameter
function get_param_index(
    x::Union{DensitySampleVector, NamedTupleDist},
    param::Integer
)
    return param
end



# construct 1D BATHistogram from samples
function BATHistogram(
    maybe_shaped_samples::DensitySampleVector,
    param::Union{Integer, Symbol};
    nbins = 200,
    closed::Symbol = :left,
    filter::Bool = false
)
    samples = BAT.unshaped.(maybe_shaped_samples)

    if filter
        samples = BAT.drop_low_weight_samples(samples)
    end

    param_idx = get_param_index(maybe_shaped_samples, param)

    hist = fit(Histogram,
            flatview(samples.v)[param_idx, :],
            FrequencyWeights(samples.weight),
            nbins = nbins, closed = closed)

    return BATHistogram(hist)
end



# construct 1D BATHistogram from prior
function BATHistogram(
    prior::NamedTupleDist,
    param::Union{Integer, Symbol};
    nbins = 200,
    closed::Symbol = :left,
    nsamples::Integer = 10^6
)
    param_idx = get_param_index(prior, param)
    r = rand(prior, nsamples)
    hist = fit(Histogram, r[param_idx, :], nbins = nbins, closed = closed)

    return BATHistogram(hist)
end



# construct 2D BATHistogram from samples
function BATHistogram(
    maybe_shaped_samples::DensitySampleVector,
    params::Union{NTuple{2, Symbol}, NTuple{2, Integer}};
    nbins = 200,
    closed::Symbol = :left,
    filter::Bool = false
)
    samples = unshaped.(maybe_shaped_samples)

    if filter
        samples = BAT.drop_low_weight_samples(samples)
    end

    param_i = get_param_index(maybe_shaped_samples, params[1])
    param_j = get_param_index(maybe_shaped_samples, params[2])

    hist = fit(Histogram,
            (flatview(samples.v)[param_i, :],
            flatview(samples.v)[param_j, :]),
            FrequencyWeights(samples.weight),
            nbins = nbins, closed = closed)

    return BATHistogram(hist)
end



# # construct 2D BATHistogram from prior
function BATHistogram(
    prior::NamedTupleDist,
    params::Union{NTuple{2, Symbol}, NTuple{2, Integer}};
    nbins = 200,
    closed::Symbol = :left,
    nsamples::Integer = 10^6
)
    param_i = get_param_index(prior, params[1])
    param_j = get_param_index(prior, params[2])

    r = rand(prior, nsamples)
    hist = fit(Histogram, (r[param_i, :], r[param_j, :]), nbins = nbins, closed = closed)

    return BATHistogram(hist)
end
