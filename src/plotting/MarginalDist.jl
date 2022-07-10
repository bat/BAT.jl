struct Marginalization{D} <: AbstractVector{D} 
    samples::DensitySampleVector
    vsel::Any
end

struct MarginalDist
    dist::Union{ReshapedDist, Distribution}
end

function _get_edges(data::Tuple, nbins::Tuple{Vararg{<:Integer}}, closed::Symbol)
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
    bins = 200, #::Union{I, Vector{I}, Tuple{I}} where I<:Integer
    closed::Symbol = :left,
    filter::Bool = false
)

    marg_samples = bat_marginalize(samples, vsel)
    vs = varshape(marg_samples)
    vs isa NamedTupleShape ? shapes = [getproperty(acc, :shape) for acc in vs._accessors] : shapes = [0,0]
    UV = (vs isa ArrayShape && vsel isa Integer) || (length(shapes) == 1 && (shapes[1] isa ScalarShape || getproperty(shapes[1], :dims) == (1,)))

    if filter
        marg_samples = BAT.drop_low_weight_samples(marg_samples)
    end

    marg_samples = flatview(unshaped.(marg_samples).v)
    cols = Tuple(Vector.(eachrow(marg_samples)))

    bins = Integer.(bins)
    edges = if isa(bins, Integer)
        _get_edges(cols, (bins,), closed)
    else
        Tuple(_get_edges(cols[i], bins[i], closed) for i in 1:length(bins))
    end

    hist = fit(Histogram, cols, edges, closed = closed)

    binned_dist = UV ? EmpiricalDistributions.UvBinnedDist(hist) : EmpiricalDistributions.MvBinnedDist(hist)
    binned_dist = UV ? binned_dist : vs(binned_dist) isa ReshapedDist ? vs(binned_dist) : ReshapedDist(vs(binned_dist), vs)

    return MarginalDist(binned_dist)
end

#for prior or MarginalDist
#TODO: think about implementing parsing of already marginalized names for remarginalization
function MarginalDist(
    dist::Union{NamedTupleDist, MarginalDist},
    vsel;
    bins = 200, #::Union{I, Vector{I}, Tuple{I}} where I<:Integer
    closed::Symbol = :left,
    nsamples::Integer = 10^6,
    filter::Bool = false
)
    dist = dist isa MarginalDist ? dist.dist : dist

    vs = varshape(dist)
    samples = vs.(DensitySampleVector(unshaped.(rand(dist, nsamples)), fill(NaN, nsamples)))

    return MarginalDist(samples, vsel; bins, closed, filter)
end
