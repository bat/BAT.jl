# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct Marginalization{D} <: AbstractVector{D}
    samples::DensitySampleVector
    vsel::Any
end


struct MarginalDist
    dist::Union{ReshapedDist, Distribution}
end


# StatsBase.histrange can allocate excessive numbers of edges when the span
# is too small to resolve the requested bins at the data's magnitude.
function _edges_would_be_degenerate(data, nbins::Integer)
    lo, hi = extrema(data)
    # Non-finite bounds invalidate the precision comparison below.
    (isfinite(lo) && isfinite(hi)) || return true
    scale = max(abs(lo), abs(hi), one(float(lo)))
    return (hi - lo) < nbins * eps(scale)
end


# Use linear edges when StatsBase's edge selection cannot resolve the bins.
function _safe_edges_fallback(data, nbins::Integer)
    lo, hi = extrema(data)
    # Keep a placeholder bin for non-finite domains; range rejects their bounds.
    (isfinite(lo) && isfinite(hi)) || return range(0.0, 1.0, length = 2)
    if lo == hi
        # Pad constant data so the bin contains samples under either closure.
        # Scale the padding to keep its endpoints distinct at large magnitudes.
        pad = max(abs(lo), one(lo)) * 0.05
        return range(lo - pad, lo + pad, length = 2)
    end

    # Extend beyond both extrema so boundary samples remain inside the bins.
    margin = (hi - lo) * 1e-3
    lower = min(lo - margin, prevfloat(float(lo)))
    upper = max(hi + margin, nextfloat(float(hi)))
    n = nbins
    # The range's extrapolated zero-crossing index can exceed Int32.
    edges = range(lower, upper, length = Int64(n) + 1)
    # Reduce resolution until rounded edges are distinct and widths agree.
    # Legacy credible-region calculations require uniform bin widths.
    # Relative tolerance avoids coarsening for ordinary roundoff at small scales.
    while !all(i -> edges[i] < edges[i + 1] &&
            isapprox(edges[i + 1] - edges[i], edges[2] - edges[1]; atol = 0),
            1:length(edges)-1)
        n = max(1, n ÷ 2)
        edges = range(lower, upper, length = Int64(n) + 1)
    end
    # Actual rounded widths can differ from the range's abstract step. Use
    # those same widths for histogram normalization and downstream rendering.
    return collect(edges)
end


function _get_edges(data::Tuple, nbins::Tuple{Vararg{Integer}}, closed::Symbol)
    counts = StatsBase._nbins_tuple(data, nbins)
    if any(_edges_would_be_degenerate(d, n) for (d, n) in zip(data, counts))
        return Tuple(_safe_edges_fallback(d, n) for (d, n) in zip(data, counts))
    end
    return StatsBase.histrange(data, counts, closed)
end


function _get_edges(data::Any, nbins::Integer, closed::Symbol)
    _edges_would_be_degenerate(data, nbins) && return _safe_edges_fallback(data, nbins)
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

    _plot_validate_weights(samples.weight)
    any(>(0), samples.weight) || _plot_weight_error("a marginal distribution requires positive mass")
    marg_samples = bat_marginalize(samples, vsel).result
    vs = varshape(marg_samples)
    vs isa NamedTupleShape ? shapes = [getproperty(acc, :shape) for acc in values(vs)] : shapes = [0,0]
    UV = (vs isa ArrayShape && vsel isa Integer) || (length(shapes) == 1 && (shapes[1] isa Union{ScalarShape, ConstValueShape} || (shapes[1] isa ArrayShape && size(shapes[1]) == (1,))))

    marg_samples = flatview(unshaped.(marg_samples).v)
    cols = Tuple(Vector.(eachrow(marg_samples)))
    weights = samples.weight
    _plot_validate_coordinates(cols, weights)
    if filter
        mask = _plot_low_weight_mask(weights)
        cols = map(v -> v[mask], cols)
        weights = view(weights, mask)
    end

    edges = if isa(bins, Integer)
        _get_edges(cols, (bins,), closed)
    elseif bins isa Tuple
        Tuple(_get_edges(cols[i], bins[i], closed) for i in 1:length(bins))
    else
        (_get_edges(cols, bins, closed),)
    end

    hist = _plot_fit_histogram(cols, weights, edges, :probability; closed=closed)

    binned_dist = _plot_binned_dist(hist, UV)
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
    samples = vs.(DensitySampleVector(v = unshaped.(rand(dist, nsamples)), logd = fill(NaN, nsamples)))

    return MarginalDist(samples, vsel; bins, closed, filter)
end
