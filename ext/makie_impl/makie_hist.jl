# This file is a part of BAT.jl, licensed under the MIT License (MIT).

function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Hist1D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return (centers=Vector{Float64}(), weights=Vector{Float64}(), widths=Vector{Float64}())
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    (; normalization, nbins, closed, filter) = config
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins + 1, closed, normalization)
    centers = _get_bin_centers(hist)
    return (centers=centers[1], weights=hist.weights, widths=collect(hist.edges[1]))
end


function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist1D,
    config::NamedTuple
)
    (; centers, weights, widths) = primitives
    (; color, alpha, filled, strokecolor, strokewidth, edge) = config

    if isempty(weights)
        return PlotSpec[]
    end

    bars = S.BarPlot(
        centers,
        weights;
        color=color,
        alpha=alpha,
        gap=0.0,
        # width=widths,
        visible=filled
    )

    stairs = S.Stairs(
        widths,
        vcat(weights, weights[end]);
        step=:post,
        color=strokecolor,
        linewidth=strokewidth,
        visible=edge
    )
    return [bars, stairs]
end

function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Hist2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return (centers_x=Vector{Float64}(), centers_y=Vector{Float64}(), weights=Matrix{Float64}(undef, 0, 0))
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    (; normalization, nbins, closed, filter) = config

    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, normalization)

    centers_x, centers_y = _get_bin_centers(hist)
    hist_weights = hist.weights
    weights = fill(NaN, size(hist_weights))
    nonzero_idxs = hist_weights .> 0
    weights[nonzero_idxs] .= hist_weights[nonzero_idxs]

    return (centers_x=centers_x, centers_y=centers_y, weights=weights)
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
    return (xy_data=Vector{Point{2,Float32}}(), widths=Vector{Float64}, stairs_data=Vector{Point{2,Float32}}(), bin_colors=Vector{RGBA{Float32}}())
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileHist1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    (; normalization, levels, colormap, alpha, rev, nbins, closed, edge, strokewidth) = config
    hist = _marginal_view_dist(marg_coords, weights, config.filter, nbins, closed, normalization)

    valid_intervals = sort(filter(x -> 0 < x < 1, levels))
    sub_hists, _ = BAT.get_smallest_intervals(hist, valid_intervals)

    pal_values = collect(range(0.4, 0.6, length(valid_intervals)))

    pal = cgrad(colormap, rev=rev, alpha=alpha)
    pal_values = collect(range(0.05, 0.7, length(valid_intervals)))
    bin_colors = fill(RGBA{Float32}(0, 0, 0, 0), length(hist.weights))

    for (i, sub_hist) in enumerate(sub_hists)
        color_idx = length(valid_intervals) - i + 1
        c = pal[pal_values[i]]
        mask = sub_hist.weights .> 0
        bin_colors[mask] .= c
    end

    centers = _get_bin_centers(hist)[1]

    xy_data = Point2f.(centers, hist.weights)
    edges = collect(hist.edges)[1]

    widths = diff(edges)

    stairs_y = vcat(hist.weights, hist.weights[end])
    stairs_data = Point2f.(edges, stairs_y)

    return (xy_data=xy_data, widths=widths, stairs_data=stairs_data, bin_colors=bin_colors)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::QuantileHist1D,
    config::NamedTuple
)
    (; xy_data, widths, stairs_data, bin_colors) = primitives
    (; edge, strokecolor, strokewidth) = config

    if isempty(widths)
        return PlotSpec[]
    end

    bars = S.BarPlot(xy_data;
        color=bin_colors,
        width=widths,
        gap=0.0,
        visible=true
    )

    final_strokewidth = edge ? strokewidth : 0.0

    stairs = S.Stairs(stairs_data;
        step=:post,
        color=strokecolor,
        linewidth=final_strokewidth,
        visible=edge
    )
    return [bars, stairs]
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::QuantileHist2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return (centers_x=Vector{Float64}(), centers_y=Vector{Float64}(), color_grid=Matrix{RGBA{Float32}}(undef, 0, 0))
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileHist2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    (; normalization, levels, colormap, alpha, rev, nbins, closed) = config
    hist = _marginal_view_dist(marg_coords, weights, config.filter, nbins, closed, normalization)

    valid_intervals = sort(filter(x -> 0 < x < 1, levels))
    sub_hists, _ = BAT.get_smallest_intervals(hist, valid_intervals)

    pal = cgrad(colormap, rev=rev, alpha=alpha)
    pal_values = collect(range(0.05, 0.7, length(valid_intervals)))

    dims = size(hist.weights)
    color_grid = fill(RGBA{Float32}(0, 0, 0, 0), dims)

    for (i, sub_hist) in enumerate(sub_hists)
        color_idx = length(valid_intervals) - i + 1
        c = pal[pal_values[i]]
        mask = sub_hist.weights .> 0
        color_grid[mask] .= c
    end

    centers_x, centers_y = _get_bin_centers(hist)

    return (centers_x=centers_x, centers_y=centers_y, color_grid=color_grid)
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
    return (x=SubArray(), y=SubArray(), weights=SubArray(), thresh=Float64())
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

    final_thresh = if isnothing(threshold)
        pos_w = weights[weights.>0]
        isempty(pos_w) ? 0.0 : minimum(pos_w)
    else
        threshold
    end

    return (x=x, y=y, weights=weights, thresh=final_thresh)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hexbin2D,
    config::NamedTuple
)
    (; x, y, weights, thresh) = primitives
    (; colormap, rev, nbins, alpha) = config
    if isempty(weights)
        return PlotSpec[]
    end
    final_cmap = rev ? Reverse(colormap) : colormap

    hex = S.Hexbin(x, y;
        weights=weights,
        bins=nbins,
        colormap=final_cmap,
        alpha=alpha,
        threshold=thresh
    )

    return [hex]
end

