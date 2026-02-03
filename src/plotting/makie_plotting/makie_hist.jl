
@recipe(Hist1D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        nbins = 30,
        closed = :left,
        normalization = :pdf,
        filter = false,
        color = Makie.wong_colors()[1],
        alpha = 1.0,
        filled = true,
        edge = false,
        strokecolor = Makie.wong_colors()[1],
        strokewidth = 1
    )
end

function Makie.plot!(p::Hist1D)
    marg_dist = lift(p.samples, p.vsel, p.nbins, p.closed, p.filter) do smpls, vsel, b, c, f
        return MarginalDist(smpls, vsel; bins=b, closed=c, filter=f)
    end

    hist_data = lift(marg_dist, p.normalization) do marg, norm
        h_raw = convert(StatsBase.Histogram, marg.dist isa BAT.ReshapedDist ? marg.dist.dist : marg.dist)
        h_norm = norm == :none ? h_raw : normalize(h_raw, mode = norm)
        centers = BAT.get_bin_centers(marg)[1]
        return (centers, h_norm.weights, h_norm.edges[1])
    end

    centers = lift(x -> x[1], hist_data)
    weights = lift(x -> x[2], hist_data)
    edges   = lift(x -> x[3], hist_data)

    barplot!(p, centers, weights;
        color = p.color,
        alpha = p.alpha,
        gap = 0.0,
        width = lift(diff, edges),
        visible = p.filled
    )

    stairs!(p, edges, lift(w -> vcat(w, w[end]), weights);
        step = :post,
        color = p.strokecolor,
        linewidth = p.strokewidth,
        visible = p.edge
    )

    return p
end


@recipe(Hist2D, x, y) do scene
    Attributes(
        weights = nothing,
        nbins = (30, 30),
        colormap = :Blues,
        alpha = 1.0,
        rev = false
    )
end

function Makie.plot!(p::Hist2D)
    hist_result = lift(p[1], p[2], p.weights, p.nbins) do x, y, w, bins

        nx, ny = bins isa Integer ? (bins, bins) : bins

        edges_x = LinRange(minimum(x), maximum(x), nx + 1)
        edges_y = LinRange(minimum(y), maximum(y), ny + 1)

        w_safe = isnothing(w) ? UnitWeights{Float64}(length(x)) : weights(w)

        h = fit(Histogram, (x, y), w_safe, (edges_x, edges_y))

        return (edges_x, edges_y, h.weights)
    end

    ex = lift(r -> r[1], hist_result)
    ey = lift(r -> r[2], hist_result)
    hw = lift(r -> r[3], hist_result)

    final_cmap = lift(p.colormap, p.rev) do cm, r
        r ? Reverse(cm) : cm
    end

    heatmap!(p, ex, ey, hw;
        colormap = final_cmap,
        alpha = p.alpha
    )

    return p
end


@recipe(QuantileHist1D, x) do scene
    Attributes(
        weights = nothing,
        nbins = 30,
        levels = cdf.(Chi(1), 0:3),
        colormap = :Blues,
        rev = false,
        alpha = 1.0,
        edge = false,
        strokecolor = Makie.wong_colors()[1],
        strokewidth = 1.0
    )
end

function Makie.plot!(p::QuantileHist1D)
    plot_data = lift(p[1], p.weights, p.nbins, p.levels, p.colormap, p.rev, p.alpha) do x, w, nbins, levels, cmap, rev, alpha

        edges = LinRange(minimum(x), maximum(x), nbins + 1)

        w_obj = isnothing(w) ? UnitWeights{Float64}(length(x)) : weights(w)
        h = fit(Histogram, x, w_obj, edges)

        p = h.weights ./ sum(h.weights)

        idx = reverse(sortperm(p))
        p_sorted = p[idx]

        c = cumsum(p_sorted)

        pal = cgrad(cmap, length(levels), categorical=true, rev=!rev, alpha=alpha)
        inds = clamp.(searchsortedlast.(Ref(levels), c), 1, length(pal))
        colors_sorted = pal[inds]

        final_colors = colors_sorted[sortperm(idx)]

        widths = diff(edges)
        mids = midpoints(edges)
        heights = p ./ widths

        return (mids, heights, widths, final_colors)
    end

    mids    = lift(d -> d[1], plot_data)
    heights = lift(d -> d[2], plot_data)
    widths  = lift(d -> d[3], plot_data)
    colors  = lift(d -> d[4], plot_data)

    final_strokewidth = lift(p.edge, p.strokewidth) do e, w
        e ? w : 0.0
    end

    barplot!(p, mids, heights;
        color = colors,
        width = widths,
        gap = 0,
        strokecolor = p.strokecolor,
        strokewidth = final_strokewidth
    )

    return p
end


@recipe(QuantileHist2D, x, y) do scene
    Attributes(
        weights = nothing,
        nbins = (30, 30),
        levels = cdf.(Chi(2), 0:3),
        colormap = :Blues,
        rev = false,
        alpha = 1.0
    )
end

function Makie.plot!(p::QuantileHist2D)
    plot_data = lift(p[1], p[2], p.weights, p.nbins, p.levels, p.colormap, p.rev, p.alpha) do x, y, w, bins, levels, cmap, rev, alpha
        nx, ny = bins isa Integer ? (bins, bins) : bins

        edges_x = LinRange(minimum(x), maximum(x), nx + 1)
        edges_y = LinRange(minimum(y), maximum(y), ny + 1)

        w_obj = isnothing(w) ? UnitWeights{Float64}(length(x)) : weights(w)
        h = fit(Histogram, (x, y), w_obj, (edges_x, edges_y))

        p_grid = h.weights ./ sum(h.weights)

        flat_p = vec(p_grid)
        idx = reverse(sortperm(flat_p))
        sorted_p = flat_p[idx]

        c = cumsum(sorted_p)

        pal = cgrad(cmap, length(levels), categorical=true, rev=!rev, alpha=alpha)
        inds = clamp.(searchsortedlast.(Ref(levels), c), 1, length(pal))
        sorted_colors = pal[inds]
        flat_colors = sorted_colors[sortperm(idx)]
        color_grid = reshape(flat_colors, (nx, ny))

        final_grid = copy(color_grid)
        final_grid[h.weights .== 0.0] .= RGBA(0, 0, 0, 0)

        return (edges_x, edges_y, final_grid)
    end

    ex = lift(d -> d[1], plot_data)
    ey = lift(d -> d[2], plot_data)
    cg = lift(d -> d[3], plot_data)

    heatmap!(p, ex, ey, cg)

    return p
end


@recipe(Hexbin2D, x, y) do scene
    Attributes(
        weights = nothing,
        nbins = (30, 30),
        colormap = :Blues,
        rev = true,
        alpha = 1.0,
        threshold = nothing
    )
end

function Makie.plot!(p::Hexbin2D)
    config = lift(p.weights, p.threshold) do w, user_thresh
        if isnothing(w)
            return (nothing, isnothing(user_thresh) ? 1.0 : user_thresh)
        else
            w_vals = w isa AbstractWeights ? w.values : w
            thresh = if isnothing(user_thresh)
                pos_w = w_vals[w_vals .> 0]
                isempty(pos_w) ? 0.0 : minimum(pos_w)
            else
                user_thresh
            end
            return (w, thresh)
        end
    end

    final_weights = lift(c -> c[1], config)
    final_thresh  = lift(c -> c[2], config)

    final_cmap = lift(p.colormap, p.rev) do cm, r
        r ? Reverse(cm) : cm
    end

    final_bins = lift(p.nbins) do b
        b isa Integer ? (b, b) : b
    end

    hexbin!(p, p[1], p[2];
        weights = final_weights,
        bins = final_bins,
        colormap = final_cmap,
        alpha = p.alpha,
        threshold = final_thresh
    )

    return p
end

