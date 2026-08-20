struct Marginalization{D} <: AbstractVector{D} 
    samples::DensitySampleVector
    vsel::Any
end

struct MarginalDist
    dist::Union{ReshapedDist, Distribution}
end

# StatsBase.histrange's "nice round number" edge-picking becomes unreliable
# once a dimension's span (max(data) - min(data)) is comparable to or smaller
# than the floating-point precision (ULP/`eps`) at that magnitude -- at that
# point the data can't even represent `nbins` distinct values, so there's no
# well-defined "nice" edge spacing to find, and the underlying algorithm's
# behavior in that regime is essentially undefined rather than merely
# imprecise. Confirmed empirically (not just suspected): a domain like
# (1e16, 1e16+1) -- an ordinary-looking, finite span, just at a large offset
# -- already returns a nonsensical few edges instead of ~100, degrading
# further (66 -> 8194 -> over a billion -> an outright hang/OutOfMemoryError)
# as the offset grows relative to the span, i.e. as fewer ULPs remain to
# resolve it. Real BAT.jl models can plausibly hit this (a parameter with a
# large natural scale, or a live domain estimate that picked up a non-finite
# sample value -- see _domain_including in the Makie extension), so this
# guards every call rather than being a narrow one-off fix.
function _edges_would_be_degenerate(data, nbins::Integer)
    lo, hi = extrema(data)
    # A non-finite bound defeats the ULP check below silently rather than
    # loudly: eps(Inf) is NaN, so `(hi - lo) < nbins * eps(scale)` evaluates
    # to `false` (every comparison against NaN is false) instead of throwing
    # or correctly flagging degeneracy -- confirmed empirically to be exactly
    # how a single Inf sample value reaching a live domain estimate reached
    # raw StatsBase.histrange and reproduced this same OOM crash a second
    # time. Treated as degenerate unconditionally: there's no well-defined
    # "nice" bin spacing over a non-finite span regardless of how large
    # nbins is.
    (isfinite(lo) && isfinite(hi)) || return true
    scale = max(abs(lo), abs(hi), one(float(lo)))
    return (hi - lo) < nbins * eps(scale)
end

# A plain linear range instead -- not "nice" round numbers, but well-defined
# and safe at any magnitude, which is what actually matters once the nice-
# number algorithm's own preconditions (enough representable precision to
# distinguish nbins+1 edges at all) no longer hold. A zero-width domain (lo
# == hi, fully degenerate) collapses to a single bin rather than dividing
# nothing into `nbins` identical edges.
function _safe_edges_fallback(data, nbins::Integer)
    lo, hi = extrema(data)
    # `range(lo, hi, ...)` itself throws ArgumentError on a non-finite bound
    # (confirmed directly) rather than silently misbehaving like histrange --
    # still guarded explicitly here so a non-finite domain degrades to an
    # (admittedly uninformative) fixed placeholder bin instead of erroring
    # out of the whole plotting callback.
    (isfinite(lo) && isfinite(hi)) || return range(0.0, 1.0, length=2)
    if lo == hi
        # range(lo, lo, length=2) -- a genuinely zero-width bin -- silently
        # drops every single sample: confirmed directly that
        # fit(Histogram, ..., closed=:left/:right) on a [lo,lo)/(lo,lo]
        # interval contains nothing, not even lo itself, regardless of how
        # many samples exist. That zeroed every bin's weight, which then
        # made StatsBase.normalize(mode=:pdf) divide 0/0 into NaN downstream
        # (see _hist1d_output etc.'s own guard against that). Padding by an
        # amount scaled to lo's own magnitude (not a fixed absolute amount)
        # keeps the padded bounds representable and genuinely distinct even
        # at extreme magnitudes, where a tiny fixed pad could round back to
        # lo exactly.
        pad = max(abs(lo), one(lo)) * 0.05
        return range(lo - pad, lo + pad, length=2)
    end
    # Small proportional margin beyond the true extrema (matching what
    # StatsBase.histrange's own "nice number" edges do implicitly, via
    # rounding up past the data): without it, a sample exactly at `hi` is
    # silently excluded from the last bin under closed=:left (this fallback
    # is only reached when the true span is already degenerate/near-eps, so
    # this matters for every sample at the boundary, not just an edge case).
    margin = (hi - lo) * 1e-3
    return range(lo - margin, hi + margin, length=nbins + 1)
end

function _get_edges(data::Tuple, nbins::Tuple{Vararg{Integer}}, closed::Symbol)
    if any(_edges_would_be_degenerate(d, n) for (d, n) in zip(data, nbins))
        return Tuple(_safe_edges_fallback(d, n) for (d, n) in zip(data, nbins))
    end
    return StatsBase.histrange(data, StatsBase._nbins_tuple(data, nbins), closed)
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

    marg_samples = bat_marginalize(samples, vsel).result
    vs = varshape(marg_samples)
    vs isa NamedTupleShape ? shapes = [getproperty(acc, :shape) for acc in vs._accessors] : shapes = [0,0]
    UV = (vs isa ArrayShape && vsel isa Integer) || (length(shapes) == 1 && (shapes[1] isa Union{ScalarShape, ConstValueShape} || getproperty(shapes[1], :dims) == (1,)))

    if filter
        marg_samples = BAT.drop_low_weight_samples(marg_samples)
    end

    marg_samples = flatview(unshaped.(marg_samples).v)
    cols = Tuple(Vector.(eachrow(marg_samples)))

    edges = if isa(bins, Integer)
        _get_edges(cols, (bins,), closed)
    elseif bins isa Tuple
        Tuple(_get_edges(cols[i], bins[i], closed) for i in 1:length(bins))
    else 
        (_get_edges(cols, bins, closed),)
    end

    hist = fit(Histogram, cols, FrequencyWeights(samples.weight), edges, closed = closed)

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
