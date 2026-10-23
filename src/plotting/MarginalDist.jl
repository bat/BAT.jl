struct Marginalization{D} <: AbstractVector{D} 
    samples::DensitySampleVector
    vsel::Any
end

# A binned distribution `dist` (`UvBinnedDist` or `MvBinnedDist`) over the
# free parameters of variates of shape `shape`:
struct MarginalDist{D<:Distribution,S<:AbstractValueShape}
    dist::D
    shape::S

    MarginalDist(dist::D; shape::S = varshape(dist)) where {D<:Distribution,S<:AbstractValueShape} = new{D,S}(dist, shape)
end

ValueShapes.varshape(marg::MarginalDist) = marg.shape

_marg_hist(marg::MarginalDist) = convert(Histogram, marg.dist)

_binned_dist(hist::Histogram{<:Real,1}) = EmpiricalDistributions.UvBinnedDist(hist)
_binned_dist(hist::Histogram) = EmpiricalDistributions.MvBinnedDist(hist)

function _get_edges(data::Tuple, nbins::Tuple{Vararg{Integer}}, closed::Symbol)
    return StatsBase.histrange(data, StatsBase._nbins_tuple(data, nbins), closed)
end

function _get_edges(data::Any, nbins::Integer, closed::Symbol)
    return StatsBase.histrange((data,), StatsBase._nbins_tuple((data,), (nbins,)), closed)[1]
end

function _get_edges(data::Any, nbins::Union{AbstractRange, Tuple{AbstractRange}}, closed::Symbol)
    return nbins
end


function MarginalDist(
    samples::Union{DensitySampleVector, StructArrays.StructVector},
    vsel;
    bins = 200,
    closed::Symbol = :left,
    filter::Bool = false
)

    if filter && !isempty(samples) && !all(iszero, samples.weight)
        samples = BAT.drop_low_weight_samples(samples)
    end

    marg_samples = bat_marginalize(samples, vsel).result
    cols = Tuple(Vector.(eachrow(flatbatch(marg_samples.v))))

    edges = if isa(bins, Integer)
        _get_edges(cols, (bins,), closed)
    elseif bins isa Tuple
        Tuple(_get_edges(cols[i], bins[i], closed) for i in 1:length(bins))
    else 
        (_get_edges(cols, bins, closed),)
    end

    hist = fit(Histogram, cols, FrequencyWeights(marg_samples.weight), edges, closed = closed)

    return MarginalDist(_binned_dist(hist), shape = varshape(marg_samples))
end

#TODO: think about implementing parsing of already marginalized names for remarginalization
function MarginalDist(
    dist::NamedTupleDist,
    vsel;
    bins = 200, #::Union{I, Vector{I}, Tuple{I}} where I<:Integer
    closed::Symbol = :left,
    nsamples::Integer = 10^6,
    filter::Bool = false
)
    μ = batmeasure(dist)
    samples = DensitySampleVector(rand(μ^nsamples), μ)
    return MarginalDist(samples, vsel; bins, closed, filter)
end

# Marginal of a marginal distribution, sums the bin probabilities over all
# parameters not selected by `vsel`:
function MarginalDist(marg::MarginalDist, vsel)
    dims, sel_shape = _resolve_vsel(varshape(marg), vsel)
    hist = _marg_hist(marg)
    masses = _bin_masses(hist)
    other_dims = Tuple(setdiff(1:ndims(masses), dims))
    marg_masses = permutedims(dropdims(sum(masses, dims = other_dims), dims = other_dims), invperm(sortperm(dims)))
    marg_hist = Histogram(hist.edges[dims], marg_masses, hist.closed, false)
    shape = varshape(marg) isa NamedTupleShape ? sel_shape : ArrayShape{Real}(length(dims))
    return MarginalDist(_binned_dist(marg_hist); shape)
end

function _bin_masses(hist::Histogram{<:Real,N}) where N
    hist.isdensity || return hist.weights
    widths = ntuple(d -> reshape(diff(hist.edges[d]), ntuple(i -> i == d ? length(hist.edges[d]) - 1 : 1, Val(N))), Val(N))
    return broadcast(*, hist.weights, widths...)
end
