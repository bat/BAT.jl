
@with_kw struct Hist1D <: BATMakieRecipe
    weights::Union{Vector{Integer},Nothing} = nothing
    nbins::Union{Tuple{Vararg{Integer}},Integer} = 30
    closed::Symbol = :left
    normalization::Symbol = :pdf
    filter::Bool = false
    color::Any = Makie.wong_colors()[1]
    alpha::Float64 = 1.0
    filled::Bool = true
    edge::Bool = false
    strokecolor::Any = Makie.wong_colors()[1]
    strokewidth::Float64 = 1.0
end

function compute_plotting_primitives(
    ::Nothing,
    ::Nothing,
    ::Hist1D
)
    return (centers=Vector{Float64}(), weights=Vector{Float64}(), widths=Vector{Float64}())
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist1D
)
    (; normalization, nbins, closed, filter) = recipe
    hist = _marginal_view_dist(marg_coords, weights, filter, nbins + 1, closed, normalization)
    centers = _get_bin_centers(hist)
    return (centers=centers[1], weights=hist.weights, widths=collect(hist.edges[1]))
end


function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist1D
)
    (; centers, weights, widths) = primitives
    (; color, alpha, filled, strokecolor, strokewidth, edge) = recipe

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


@with_kw struct Hist2D <: BATMakieRecipe
    weights::Union{Vector{Integer},Nothing} = nothing
    nbins::Union{Tuple{Vararg{Integer}},Integer} = (30, 30)
    closed::Symbol = :left
    normalization::Symbol = :pdf
    filter::Bool = false
    colormap::Symbol = :Blues
    alpha::Float64 = 1.0
    rev::Bool = false
end

function compute_plotting_primitives(
    ::Nothing,
    ::Nothing,
    ::Hist2D
)
    return (centers=Matrix{Float64}(), weights=Vector{Float64}())
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Hist2D
)
    (; normalization, nbins, closed, filter) = recipe

    hist = _marginal_view_dist(marg_coords, weights, filter, nbins, closed, normalization)

    centers = _get_bin_centers(hist)

    return (centers=centers, weights=hist.weights)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Hist2D
)
    (; centers, weights) = primitives
    (; colormap, alpha, rev) = recipe
    final_cmap = rev ? Reverse(colormap) : colormap

    heat = S.Heatmap(
        centers[1], centers[2], weights;
        colormap=final_cmap,
        alpha=alpha
    )
    return [heat]
end


@with_kw struct QuantileHist1D <: BATMakieRecipe
    weights = nothing
    nbins = 30
    closed = :left
    normalization = :pdf
    filter = false
    levels = cdf.(Chi(1), 0:3)
    colormap = :Blues
    rev = false
    alpha = 1.0
    edge = false
    strokecolor = Makie.wong_colors()[1]
    strokewidth = 1.0
end

function compute_plotting_primitives(
    ::Nothing,
    ::Nothing,
    ::QuantileHist1D
)
    return (xy_data=Vector{Point2f}(), widths=Vector{Float64}, stairs_data=Vector{Point2f}(), bin_colors=Vector{RGBA}())
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileHist1D
)
    (; normalization, levels, colormap, alpha, rev, nbins, closed, edge, strokewidth) = recipe
    hist = _marginal_view_dist(marg_coords, weights, recipe.filter, nbins, closed, normalization)

    valid_intervals = sort(filter(x -> 0 < x < 1, levels))
    sub_hists, _ = BAT.get_smallest_intervals(hist, valid_intervals)

    pal = cgrad(colormap, length(valid_intervals), categorical=true, rev=!rev, alpha=alpha)
    bin_colors = fill(RGBA{Float32}(0, 0, 0, 0), length(hist.weights))

    for (i, sub_hist) in enumerate(sub_hists)
        color_idx = length(valid_intervals) - i + 1
        c = pal[color_idx]
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
    recipe::QuantileHist1D
)
    (; xy_data, widths, stairs_data, bin_colors) = primitives
    (; edge, strokecolor, strokewidth) = recipe

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



@with_kw struct QuantileHist2D <: BATMakieRecipe
    weights = nothing
    nbins = (30, 30)
    closed = :left
    normalization = :pdf
    filter = false
    levels = cdf.(Chi(2), 0:3)
    colormap = :Blues
    rev = false
    alpha = 1.0
end

function compute_plotting_primitives(
    ::Nothing,
    ::Nothing,
    ::QuantileHist2D
)
    return (1,)
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileHist2D
)

    (; normalization, levels, colormap, alpha, rev, nbins, closed) = recipe
    hist = _marginal_view_dist(marg_coords, weights, recipe.filter, nbins, closed, normalization)

    valid_intervals = sort(filter(x -> 0 < x < 1, levels))
    sub_hists, _ = BAT.get_smallest_intervals(hist, valid_intervals)

    pal = cgrad(colormap, length(valid_intervals), categorical=true, rev=!rev, alpha=alpha)
    dims = size(hist.weights)
    color_grid = fill(RGBA{Float32}(0, 0, 0, 0), dims)

    for (i, sub_hist) in enumerate(sub_hists)
        color_idx = length(valid_intervals) - i + 1
        c = pal[color_idx]
        mask = sub_hist.weights .> 0
        color_grid[mask] .= c
    end

    centers = _get_bin_centers(hist)

    return (centers=centers, color_grid=color_grid)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::QuantileHist2D
)
    (; centers, color_grid) = primitives
    heat = S.Heatmap(centers[1], centers[2], color_grid)
    return [heat]
end



@with_kw struct Hexbin2D <: BATMakieRecipe
    weights = nothing
    nbins = (30, 30)
    filter = false
    colormap = :Blues
    rev = true
    alpha = 1.0
    threshold = nothing
end

function Makie.plot!(p::Hexbin2D)
    data = lift(p.samples, p.vsel, p.filter) do smpls, vsel, f
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = BAT.drop_low_weight_samples(marg_res)
        end

        w = marg_res.weight

        flat_vals = flatview(unshaped.(marg_res).v)
        x = flat_vals[1, :]
        y = flat_vals[2, :]

        return (x, y, w)
    end

    x_vals = lift(d -> d[1], data)
    y_vals = lift(d -> d[2], data)
    w_vals = lift(d -> d[3], data)

    final_thresh = lift(w_vals, p.threshold) do w, user_thresh
        thresh = if isnothing(user_thresh)
            pos_w = w[w.>0]
            isempty(pos_w) ? 0.0 : minimum(pos_w)
        else
            user_thresh
        end
        return thresh
    end

    final_cmap = lift(p.colormap, p.rev) do cm, r
        r ? Reverse(cm) : cm
    end

    final_bins = lift(p.nbins) do b
        b isa Integer ? (b, b) : b
    end

    hexbin!(p, x_vals, y_vals;
        weights=w_vals,
        bins=final_bins,
        colormap=final_cmap,
        alpha=p.alpha,
        threshold=final_thresh
    )

    return p
end

