# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Fresh sentinels per call, not shared consts -- see _empty_scatter2d_primitives (makie_scatter.jl).
_empty_hist1d_primitives() = (centers=Vector{Float64}(), weights=Vector{Float64}(), edges=Vector{Float64}())
_empty_hist2d_primitives() = (centers_x=Vector{Float64}(), centers_y=Vector{Float64}(), weights=Matrix{Float64}(undef, 0, 0))
_empty_quantilehist1d_primitives() = (xy_data=Vector{Point{2,Float32}}(), widths=Vector{Float64}(), stairs_data=Vector{Point{2,Float32}}(), bin_colors=Vector{RGBA{Float32}}())
_empty_quantilehist2d_primitives() = (centers_x=Vector{Float64}(), centers_y=Vector{Float64}(), color_grid=Matrix{RGBA{Float32}}(undef, 0, 0))
_empty_hexbin2d_primitives() = (x=Float64[], y=Float64[], weights=Float64[], thresh=0.0)

# Bin edges for a marginal histogram: derived from the fixed per-dimension
# `domain` when available (keeps live bins stable as samples accumulate),
# data-derived otherwise.
function _hist_edges(::Nothing, cols, bins::Integer, closed::Symbol)
    return _get_edges(cols, (bins,), closed)
end
function _hist_edges(::Nothing, cols, bins::Tuple, closed::Symbol)
    return Tuple(_get_edges(cols[i], bins[i], closed) for i in 1:length(bins))
end
function _hist_edges(domain::Tuple{Float64,Float64}, cols, bins::Integer, closed::Symbol)
    return _get_edges(([domain[1], domain[2]],), (bins,), closed)
end
function _hist_edges(domain::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}, cols, bins::Tuple, closed::Symbol)
    (dlo1, dhi1), (dlo2, dhi2) = domain
    return _get_edges(([dlo1, dhi1], [dlo2, dhi2]), bins, closed)
end

function _marginal_view_dist(
    locations::SubArray,
    weights::SubArray,
    filter::Bool,
    bins::Union{Tuple{Vararg{Int64}},Int64},
    closed::Symbol,
    normalization::Symbol,
    domain = nothing,
)
    if filter
        mask = _low_weight_mask(weights)
        locations = view(locations, :, mask)
        weights = view(weights, mask)
    end

    cols = Tuple(eachrow(locations))
    edges = _hist_edges(domain, cols, bins, closed)

    hist = fit(Histogram, cols, FrequencyWeights(weights), edges, closed=closed)
    h_norm = normalization == :none ? hist : StatsBase.normalize(hist, mode=normalization)
    return h_norm
end

# Mirrors BAT.drop_low_weight_samples, but returns a mask over a bare weight vector.
function _low_weight_mask(weights::AbstractVector, fraction::Real = 10^-5, threshold::Real = 10^-2)
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

    # Explicitly typed comprehensions: an untyped comprehension over <= 1 edges
    # infers Vector{Any}, violating the graph's TypedEdge live/dead type contract.
    centers = Vector{Float64}[
        Float64[edges[d][i] + 0.5 * (edges[d][i+1] - edges[d][i]) for i in 1:length(edges[d])-1]
        for d in 1:dims
    ]

    return centers
end

# Normalizing an all-zero-weight histogram gives 0/0 = NaN bins, which would
# corrupt the shared diagonal y-axis limit -- skip normalization instead.
_safe_normalize(hist::Histogram, normalization::Symbol) =
    (normalization == :none || iszero(sum(hist.weights))) ? hist : StatsBase.normalize(hist, mode=normalization)

function _hist1d_output(hist::Histogram, normalization::Symbol)
    h_norm = _safe_normalize(hist, normalization)
    centers = _get_bin_centers(h_norm)
    # Float64.(...): with :none normalization the weights keep the samples' own
    # eltype, mismatching the empty sentinel's Vector{Float64} (TypedEdge).
    return (centers=centers[1], weights=Float64.(h_norm.weights), edges=collect(h_norm.edges[1]))
end

function _hist2d_output(hist::Histogram, normalization::Symbol)
    h_norm = _safe_normalize(hist, normalization)
    centers_x, centers_y = _get_bin_centers(h_norm)
    # Zero-count bins as NaN (rendered transparent); Float64 forced since the
    # all-zero-weight case skips normalization and keeps the integer eltype.
    weights = map(w -> w > 0 ? Float64(w) : NaN, h_norm.weights)
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
    edges = h_norm.edges[1]
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

function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Hist1D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_hist1d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    # A live cell can still have zero samples -- degrade like a dead cell.
    isempty(weights) && return _empty_hist1d_primitives()
    (; normalization, nbins, closed, filter) = config
    # nbins + 1, unlike every other hist recipe's plain nbins -- deliberate asymmetry.
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins + 1, closed, :none, get(config, :domain, nothing))
    return _hist1d_output(hist, normalization)
end


function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist1D,
    config::NamedTuple
)
    (; centers, weights, edges) = primitives

    if isempty(weights)
        return PlotSpec[]
    end

    bars = S.BarPlot(centers, weights)
    stairs = S.Stairs(edges, vcat(weights, weights[end]))

    return [bars, stairs]
end

# Peak y-value of a diagonal recipe's primitives -- _init_gridlayout links all
# diagonal cells to a shared y range; the generic fallback covers stats-only recipes.
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
    return _empty_hist2d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    isempty(weights) && return _empty_hist2d_primitives()
    (; normalization, nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, :none, get(config, :domain, nothing))
    return _hist2d_output(hist, normalization)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; centers_x, centers_y, weights) = primitives

    if isempty(weights)
        return PlotSpec[]
    end

    # Lower-triangle cells swap axes at compose time (see _init_gridlayout's
    # invariant comment). permutedims, not lazy transpose: recursive transpose has
    # no method for the RGBA cells QuantileHist2D/QuantileKDE2D feed through here.
    heat = transposed ?
        S.Heatmap(centers_y, centers_x, permutedims(weights)) :
        S.Heatmap(centers_x, centers_y, weights)
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
    return _empty_quantilehist1d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileHist1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    isempty(weights) && return _empty_quantilehist1d_primitives()
    (; nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, :none, get(config, :domain, nothing))
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
    return _empty_quantilehist2d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileHist2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    isempty(weights) && return _empty_quantilehist2d_primitives()
    (; nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, :none, get(config, :domain, nothing))
    return _quantilehist2d_output(hist, config)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::QuantileHist2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; centers_x, centers_y, color_grid) = primitives
    if isempty(centers_x)
        return PlotSpec[]
    end
    # permutedims (not transpose) -- see Hist2D's matching comment above.
    heat = transposed ?
        S.Heatmap(centers_y, centers_x, permutedims(color_grid)) :
        S.Heatmap(centers_x, centers_y, color_grid)
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
    return _empty_hexbin2d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hexbin2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    # Zero-sample live cell: the placeholder view is 0x0, so row indexing would throw.
    isempty(weights) && return _empty_hexbin2d_primitives()
    (; threshold) = config
    x = marg_coords[1, :]
    y = marg_coords[2, :]

    # Float64(...): match _empty_hexbin2d_primitives()'s thresh::Float64.
    final_thresh = if isnothing(threshold)
        pos_w = weights[weights.>0]
        isempty(pos_w) ? 0.0 : Float64(minimum(pos_w))
    else
        threshold
    end

    # weights materialized to Vector{Float64} -- see Scatter2D's matching comment
    # (makie_scatter.jl) on the live/dead TypedEdge type-lock.
    return (x=x, y=y, weights=Float64.(weights), thresh=final_thresh)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hexbin2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; x, y, weights, thresh) = primitives
    (; colormap, rev, nbins) = config
    if isempty(weights)
        return PlotSpec[]
    end
    final_cmap = rev ? Reverse(colormap) : colormap

    # See Scatter2D's matching comment (makie_scatter.jl); per-axis bin counts
    # swap along with the axes.
    if transposed
        x, y = y, x
        nbins isa Tuple && (nbins = reverse(nbins))
    end

    hex = S.Hexbin(x, y;
        weights=weights,
        bins=nbins,
        colormap=final_cmap,
        threshold=thresh
    )

    return [hex]
end

