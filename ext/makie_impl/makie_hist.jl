# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Dead-cell results are always identical and read-only (isempty-checked, then
# discarded), so these are shared const sentinels instead of fresh allocations
# on every recompute (which happens on every new sample batch for every
# non-selected recipe).
const _EMPTY_HIST1D_PRIMITIVES = (centers=Vector{Float64}(), weights=Vector{Float64}(), widths=Vector{Float64}())
const _EMPTY_HIST2D_PRIMITIVES = (centers_x=Vector{Float64}(), centers_y=Vector{Float64}(), weights=Matrix{Float64}(undef, 0, 0))
const _EMPTY_QUANTILEHIST1D_PRIMITIVES = (xy_data=Vector{Point{2,Float32}}(), widths=Vector{Float64}(), stairs_data=Vector{Point{2,Float32}}(), bin_colors=Vector{RGBA{Float32}}())
const _EMPTY_QUANTILEHIST2D_PRIMITIVES = (centers_x=Vector{Float64}(), centers_y=Vector{Float64}(), color_grid=Matrix{RGBA{Float32}}(undef, 0, 0))
const _EMPTY_HEXBIN2D_PRIMITIVES = (x=Float64[], y=Float64[], weights=Float64[], thresh=0.0)

# Hist1D/Hist2D/QuantileHist1D/QuantileHist2D are updated incrementally from a
# running Histogram with fixed (pre-estimated) bin edges instead of being
# refit from the full accumulated sample set on every batch -- see
# _IncrementalHist1DState/2DState below. Not compatible with filter=true: the
# low-weight cutoff depends on the *global* weight distribution recomputed
# retroactively on every call, which doesn't fit an incremental model --
# callers must check `!config.filter` themselves alongside this trait (kept
# separate since is_incremental itself only reflects the recipe, not a
# per-call config choice).
is_incremental(::Union{Hist1D,Hist2D,QuantileHist1D,QuantileHist2D}) = true

# Running (fixed-edge) histogram state for the incremental recipes above.
# Reset (full rebuild, not merge) if the real variable(s) this grid cell
# represents changes (a vsel change), the fixed domain the edges were built
# from widens (see makie_visualizer.jl's overflow handling), the sample count
# goes backwards (e.g. buffered samples get cleared), or the window's start
# position moves (the index-range slider's low handle -- see wstart's comment
# on _update_hist! below).
mutable struct _IncrementalHist1DState
    hist::Union{Histogram,Nothing}
    n::Int
    vsel::Int       # real variable index this state currently tracks (0 = none yet)
    domain::Tuple{Float64,Float64}
    wstart::Int     # window_start this state was last built/updated against
end
_IncrementalHist1DState() = _IncrementalHist1DState(nothing, 0, 0, (0.0, 0.0), 1)

mutable struct _IncrementalHist2DState
    hist::Union{Histogram,Nothing}
    n::Int
    vsel::Tuple{Int,Int}
    domain::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}
    wstart::Int
end
_IncrementalHist2DState() = _IncrementalHist2DState(nothing, 0, (0, 0), ((0.0, 0.0), (0.0, 0.0)), 1)

# Folds only the newly-arrived samples (coords[state.n+1:end]) into the
# running histogram (fit over just the new slice, then merge!'d into the
# existing one -- both share the same fixed edges), rebuilding from scratch
# over the full current view instead if vsel/domain changed or data shrank.
#
# `wstart` (the index-range slider's low end, see :window_start in
# makie_compute_graph.jl) must also force a rebuild, not just a shrink in
# `n`: this state's only signal for "how much of `coords` is genuinely new"
# is length, which silently breaks once `coords` can be a *panned* window
# instead of always growing from a fixed start -- shifting the whole
# interval (both slider handles at once) keeps the window's width, and
# therefore `n`, exactly constant while every sample in it changes. Without
# this check that shift is invisible to the `n_now > state.n` branch below,
# so the plot would silently keep showing the stale pre-shift histogram
# forever (confirmed via direct repro: shifting a fixed-width window left/
# right produced bit-identical `weights` output despite `coords` itself
# genuinely differing).
function _update_hist!(state::_IncrementalHist1DState, coords::AbstractVector, weights::AbstractVector, vsel::Int, domain::Tuple{Float64,Float64}, nbins::Integer, closed::Symbol, wstart::Integer)
    n_now = length(coords)
    if isnothing(state.hist) || state.vsel != vsel || state.domain != domain || state.wstart != wstart || state.n > n_now
        edges = _get_edges(([domain[1], domain[2]],), (nbins,), closed)
        state.hist = fit(Histogram, (coords,), FrequencyWeights(weights), edges, closed=closed)
        state.n, state.vsel, state.domain, state.wstart = n_now, vsel, domain, wstart
    elseif n_now > state.n
        rng = (state.n+1):n_now
        new_hist = fit(Histogram, (view(coords, rng),), FrequencyWeights(view(weights, rng)), state.hist.edges, closed=closed)
        merge!(state.hist, new_hist)
        state.n = n_now
    end
    return state.hist
end

# coords/x/y here are full 2xN (or separate x,y) views over the *entire*
# current sample range, matching the 1D case's convention above. See the 1D
# method's comment for why `wstart` must also force a rebuild.
function _update_hist!(state::_IncrementalHist2DState, x::AbstractVector, y::AbstractVector, weights::AbstractVector, vsel::Tuple{Int,Int}, domain::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}, nbins::Tuple{Int,Int}, closed::Symbol, wstart::Integer)
    n_now = length(weights)
    if isnothing(state.hist) || state.vsel != vsel || state.domain != domain || state.wstart != wstart || state.n > n_now
        (dlo1, dhi1), (dlo2, dhi2) = domain
        edges = _get_edges(([dlo1, dhi1], [dlo2, dhi2]), nbins, closed)
        state.hist = fit(Histogram, (x, y), FrequencyWeights(weights), edges, closed=closed)
        state.n, state.vsel, state.domain, state.wstart = n_now, vsel, domain, wstart
    elseif n_now > state.n
        rng = (state.n+1):n_now
        new_hist = fit(Histogram, (view(x, rng), view(y, rng)), FrequencyWeights(view(weights, rng)), state.hist.edges, closed=closed)
        merge!(state.hist, new_hist)
        state.n = n_now
    end
    return state.hist
end

function _marginal_view_dist(
    locations::SubArray,
    weights::SubArray,
    filter::Bool,
    bins::Union{Tuple{Vararg{Int64}},Int64},
    closed::Symbol,
    normalization::Symbol
)
    if filter
        mask = _low_weight_mask(weights)
        locations = view(locations, :, mask)
        weights = view(weights, mask)
    end

    cols = Tuple(eachrow(locations))
    edges = if isa(bins, Integer)
        _get_edges(cols, (bins,), closed)
    elseif bins isa Tuple
        Tuple(_get_edges(cols[i], bins[i], closed) for i in 1:length(bins))
    else
        (_get_edges(cols, bins, closed),)
    end

    hist = fit(Histogram, cols, FrequencyWeights(weights), edges, closed=closed)
    h_norm = normalization == :none ? hist : StatsBase.normalize(hist, mode=normalization)
    return h_norm
end

# Mirrors BAT.drop_low_weight_samples, but returns a mask over a bare weight
# vector instead of indexing a DensitySampleVector (locations/weights are
# already split apart into separate views by the time they get here).
function _low_weight_mask(weights::AbstractVector, fraction::Real=10^-5, threshold::Real=10^-2)
    W = float(weights)
    if minimum(W) / maximum(W) > threshold
        return trues(length(W))
    end
    W_s = sort(W)
    Q = cumsum(W_s)
    Q ./= maximum(Q)
    ind = searchsortedlast(Q, fraction)
    ind == 0 && return trues(length(W))
    thresh = W_s[ind]
    return W .>= thresh
end

function _get_bin_centers(hist::Histogram)
    edges = hist.edges
    dims = ndims(hist.weights)

    centers = [[edges[d][i] + 0.5 * (edges[d][i+1] - edges[d][i]) for i in 1:length(edges[d])-1] for d in 1:dims]

    return centers
end

# Shared postprocessing (normalize + derive plot primitives) between the
# incremental path above and the full-recompute path below, so the two only
# differ in *how* the raw Histogram gets built, not in what happens to it
# afterwards.
# StatsBase.normalize(hist, mode=:pdf) divides every bin by the total
# weight -- 0/0 = NaN when that total is exactly zero (either genuinely
# all-zero-weight samples, or -- before _safe_edges_fallback's own fix --
# a degenerate domain whose zero-width bin silently dropped every sample).
# That NaN then propagates into _diag_y_extent/diag_y_max (makie_gridlayout.jl),
# silently corrupting the shared diagonal y-axis limit for every diagonal
# cell, not just this one. Skipping normalization when the total is zero
# leaves a real, all-zero (not NaN) histogram instead -- renders as empty/
# flat bars, matching what an actually-empty dataset looks like elsewhere in
# this codebase, rather than corrupting shared state.
_safe_normalize(hist::Histogram, normalization::Symbol) =
    (normalization == :none || iszero(sum(hist.weights))) ? hist : StatsBase.normalize(hist, mode=normalization)

function _hist1d_output(hist::Histogram, normalization::Symbol)
    h_norm = _safe_normalize(hist, normalization)
    centers = _get_bin_centers(h_norm)
    # Float64.(...): h_norm.weights only gets promoted to Float64 by
    # StatsBase.normalize when normalization != :none -- with :none (currently
    # unreachable, normalization is hardcoded to :pdf everywhere in this
    # codebase) it would keep the input samples' own weight eltype (Int64 by
    # default), mismatching _EMPTY_HIST1D_PRIMITIVES's declared Vector{Float64}
    # the same way KDE1D/KDE2D/QuantileKDE2D's StepRangeLen mismatch did.
    return (centers=centers[1], weights=Float64.(h_norm.weights), widths=collect(h_norm.edges[1]))
end

function _hist2d_output(hist::Histogram, normalization::Symbol)
    h_norm = _safe_normalize(hist, normalization)
    centers_x, centers_y = _get_bin_centers(h_norm)
    hist_weights = h_norm.weights
    weights = fill(NaN, size(hist_weights))
    nonzero_idxs = hist_weights .> 0
    weights[nonzero_idxs] .= hist_weights[nonzero_idxs]
    return (centers_x=centers_x, centers_y=centers_y, weights=weights)
end

function _quantilehist1d_output(hist::Histogram, config::NamedTuple)
    (; normalization, levels) = config
    h_norm = _safe_normalize(hist, normalization)

    valid_intervals = sort(filter(x -> 0 < x < 1, levels))
    sub_hists, _ = BAT.get_smallest_intervals(h_norm, valid_intervals)

    bin_colors = fill(RGBA{Float32}(0, 0, 0, 0), length(h_norm.weights))

    for (i, sub_hist) in enumerate(sub_hists)
        c = _quantile_level_color(i)
        mask = sub_hist.weights .> 0
        bin_colors[mask] .= c
    end

    centers = _get_bin_centers(h_norm)[1]
    xy_data = Point2f.(centers, h_norm.weights)
    edges = collect(h_norm.edges)[1]
    widths = diff(edges)
    stairs_y = vcat(h_norm.weights, h_norm.weights[end])
    stairs_data = Point2f.(edges, stairs_y)

    return (xy_data=xy_data, widths=widths, stairs_data=stairs_data, bin_colors=bin_colors)
end

function _quantilehist2d_output(hist::Histogram, config::NamedTuple)
    (; normalization, levels) = config
    h_norm = _safe_normalize(hist, normalization)

    valid_intervals = sort(filter(x -> 0 < x < 1, levels))
    sub_hists, _ = BAT.get_smallest_intervals(h_norm, valid_intervals)

    dims = size(h_norm.weights)
    color_grid = fill(RGBA{Float32}(0, 0, 0, 0), dims)

    for (i, sub_hist) in enumerate(sub_hists)
        c = _quantile_level_color(i)
        mask = sub_hist.weights .> 0
        color_grid[mask] .= c
    end

    centers_x, centers_y = _get_bin_centers(h_norm)
    return (centers_x=centers_x, centers_y=centers_y, color_grid=color_grid)
end

function compute_hist_primitives(::Hist1D, state::_IncrementalHist1DState, config::NamedTuple)
    isnothing(state.hist) && return _EMPTY_HIST1D_PRIMITIVES
    return _hist1d_output(state.hist, config.normalization)
end

function compute_hist_primitives(::Hist2D, state::_IncrementalHist2DState, config::NamedTuple)
    isnothing(state.hist) && return _EMPTY_HIST2D_PRIMITIVES
    return _hist2d_output(state.hist, config.normalization)
end

function compute_hist_primitives(::QuantileHist1D, state::_IncrementalHist1DState, config::NamedTuple)
    isnothing(state.hist) && return _EMPTY_QUANTILEHIST1D_PRIMITIVES
    return _quantilehist1d_output(state.hist, config)
end

function compute_hist_primitives(::QuantileHist2D, state::_IncrementalHist2DState, config::NamedTuple)
    isnothing(state.hist) && return _EMPTY_QUANTILEHIST2D_PRIMITIVES
    return _quantilehist2d_output(state.hist, config)
end

function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Hist1D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _EMPTY_HIST1D_PRIMITIVES
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    # A live cell can still have zero samples (e.g. right after vsel activates,
    # before the first batch flushes, or if buffered samples get cleared later)
    # -- degrade to the same empty result as a dead cell rather than crashing.
    isempty(weights) && return _EMPTY_HIST1D_PRIMITIVES
    (; normalization, nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins + 1, closed, :none)
    return _hist1d_output(hist, normalization)
end


function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist1D,
    config::NamedTuple
)
    (; centers, weights, widths) = primitives

    if isempty(weights)
        return PlotSpec[]
    end

    bars = S.BarPlot(centers, weights)
    stairs = S.Stairs(widths, vcat(weights, weights[end]))

    return [bars, stairs]
end

# Peak y-value of this diagonal recipe's own primitives (density/bar height),
# used by _init_gridlayout to link all diagonal cells' y-axis limits to a
# shared (0, 1.1*max) range instead of each auto-scaling independently.
# Specific methods below cover the recipes selectable as the diagonal recipe
# (see options1D in _build_fig); this generic fallback covers the rest
# (Std1D/Mean1D/Errorbars1D -- stats-overlay-only recipes that can still reach
# here as `diagonal_recipe`, e.g. via the precompile workload) with 0.0, since
# they're VLines/errorbars with no "peak density" of their own.
_diag_y_extent(::NamedTuple, ::BATMakieRecipe) = 0.0
_diag_y_extent(primitives::NamedTuple, ::Hist1D) = isempty(primitives.weights) ? 0.0 : maximum(primitives.weights)

function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Hist2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _EMPTY_HIST2D_PRIMITIVES
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    isempty(weights) && return _EMPTY_HIST2D_PRIMITIVES
    (; normalization, nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, :none)
    return _hist2d_output(hist, normalization)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist2D,
    config::NamedTuple
)
    (; centers_x, centers_y, weights) = primitives

    if isempty(weights)
        return PlotSpec[]
    end

    heat = S.Heatmap(centers_x, centers_y, weights)
    return [heat]
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::QuantileHist1D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _EMPTY_QUANTILEHIST1D_PRIMITIVES
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileHist1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    isempty(weights) && return _EMPTY_QUANTILEHIST1D_PRIMITIVES
    (; nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, :none)
    return _quantilehist1d_output(hist, config)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::QuantileHist1D,
    config::NamedTuple
)
    (; xy_data, widths, stairs_data, bin_colors) = primitives

    if isempty(widths)
        return PlotSpec[]
    end

    bars = S.BarPlot(xy_data;
        color=bin_colors,
        width=widths,
    )

    stairs = S.Stairs(stairs_data)

    return [bars, stairs]
end

_diag_y_extent(primitives::NamedTuple, ::QuantileHist1D) = isempty(primitives.xy_data) ? 0.0 : maximum(p -> p[2], primitives.xy_data)


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::QuantileHist2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _EMPTY_QUANTILEHIST2D_PRIMITIVES
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileHist2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    isempty(weights) && return _EMPTY_QUANTILEHIST2D_PRIMITIVES
    (; nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, :none)
    return _quantilehist2d_output(hist, config)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::QuantileHist2D,
    config::NamedTuple
)
    (; centers_x, centers_y, color_grid) = primitives
    if isempty(centers_x)
        return PlotSpec[]
    end
    heat = S.Heatmap(centers_x, centers_y, color_grid)
    return [heat]
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Hexbin2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _EMPTY_HEXBIN2D_PRIMITIVES
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hexbin2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    (; threshold) = config
    x = marg_coords[1, :]
    y = marg_coords[2, :]

    # Float64(...): minimum(pos_w) otherwise inherits the samples' own weight
    # eltype (Int64 by default) rather than matching
    # _EMPTY_HEXBIN2D_PRIMITIVES's declared thresh::Float64 -- harmless today
    # (scalar Int64->Float64 always converts, unlike the Vector/Range
    # mismatches this same pattern guards against elsewhere in this file) but
    # kept consistent defensively.
    final_thresh = if isnothing(threshold)
        pos_w = weights[weights.>0]
        isempty(pos_w) ? 0.0 : Float64(minimum(pos_w))
    else
        threshold
    end

    # Materialized to Vector{Float64} rather than passed through as the raw
    # SubArray{<:Any} from :flat_weights -- ComputePipeline's TypedEdge fixes
    # this node's output type from its first resolution, and the dead-cell
    # fallback (_EMPTY_HEXBIN2D_PRIMITIVES) hardcodes weights=Float64[]; a
    # live SubArray{Int64,...} (or any non-Float64 concrete type) here would
    # make a later live->dead transition (e.g. via vsel reduction) fail to
    # convert.
    return (x=x, y=y, weights=Float64.(weights), thresh=final_thresh)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hexbin2D,
    config::NamedTuple
)
    (; x, y, weights, thresh) = primitives
    (; colormap, rev, nbins) = config
    if isempty(weights)
        return PlotSpec[]
    end
    final_cmap = rev ? Reverse(colormap) : colormap

    hex = S.Hexbin(x, y;
        weights=weights,
        bins=nbins,
        colormap=final_cmap,
        threshold=thresh
    )

    return [hex]
end

