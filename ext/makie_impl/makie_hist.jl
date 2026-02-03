
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
        d = marg.dist isa BAT.ReshapedDist ? marg.dist.dist : marg.dist
        h_raw = convert(StatsBase.Histogram, d)
        h_norm = norm == :none ? h_raw : StatsBase.normalize(h_raw, mode = norm)
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


@recipe(Hist2D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        nbins = (30, 30),
        closed = :left,
        normalization = :pdf,
        filter = false,
        colormap = :Blues,
        alpha = 1.0,
        rev = false
    )
end

function Makie.plot!(p::Hist2D)
    marg_dist = lift(p.samples, p.vsel, p.nbins, p.closed, p.filter) do smpls, vsel, b, c, f
        return MarginalDist(smpls, vsel; bins=b, closed=c, filter=f)
    end

    plot_data = lift(marg_dist, p.normalization) do marg, norm
        d = marg.dist isa BAT.ReshapedDist ? marg.dist.dist : marg.dist
        h_raw = convert(StatsBase.Histogram, d)
        h_norm = norm == :none ? h_raw : StatsBase.normalize(h_raw, mode = norm)
        centers = BAT.get_bin_centers(marg)
        return (centers[1], centers[2], h_norm.weights)
    end

    x_centers = lift(x -> x[1], plot_data)
    y_centers = lift(x -> x[2], plot_data)
    weights   = lift(x -> x[3], plot_data)

    final_cmap = lift(p.colormap, p.rev) do cm, r
        r ? Reverse(cm) : cm
    end

    heatmap!(p, x_centers, y_centers, weights;
        colormap = final_cmap,
        alpha = p.alpha
    )
    return p
end

@recipe(QuantileHist1D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        nbins = 30,
        closed = :left,
        normalization = :pdf,
        filter = false,
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
    marg_dist = lift(p.samples, p.vsel, p.nbins, p.closed, p.filter) do smpls, vsel, b, c, f
        return MarginalDist(smpls, vsel; bins=b, closed=c, filter=f)
    end

    plot_data = lift(marg_dist, p.normalization, p.levels, p.colormap, p.rev, p.alpha) do marg, norm, levels, cmap, rev, alpha
        d = marg.dist isa BAT.ReshapedDist ? marg.dist.dist : marg.dist
        h_raw = convert(StatsBase.Histogram, d)
        h_norm = norm == :none ? h_raw : StatsBase.normalize(h_raw, mode = norm)

        valid_intervals = sort(filter(x -> 0 < x < 1, levels))
        hists, _ = BAT.get_smallest_intervals(h_norm, valid_intervals)
        pal = cgrad(cmap, length(valid_intervals), categorical=true, rev=!rev, alpha=alpha)
        n_bins = length(h_norm.weights)
        bin_colors = fill(RGBA{Float32}(0,0,0,0), n_bins)

        for (i, sub_hist) in enumerate(hists)
            color_idx = length(valid_intervals) - i + 1
            c = pal[color_idx]
            mask = sub_hist.weights .> 0
            bin_colors[mask] .= c
        end

        centers = BAT.get_bin_centers(marg)[1]
        return (centers, h_norm.weights, h_norm.edges[1], bin_colors)
    end

    centers = lift(d -> d[1], plot_data)
    weights = lift(d -> d[2], plot_data)
    edges   = lift(d -> d[3], plot_data)
    colors  = lift(d -> d[4], plot_data)

    barplot!(p, centers, weights;
        color = colors,
        width = lift(diff, edges),
        gap = 0.0,
        visible = true
    )

    final_strokewidth = lift(p.edge, p.strokewidth) do e, w
        e ? w : 0.0
    end

    stairs!(p, edges, lift(w -> vcat(w, w[end]), weights);
        step = :post,
        color = p.strokecolor,
        linewidth = final_strokewidth,
        visible = p.edge
    )

    return p
end


@recipe(QuantileHist2D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        nbins = (30,30),
        closed = :left,
        normalization = :pdf,
        filter = false,
        levels = cdf.(Chi(2), 0:3),
        colormap = :Blues,
        rev = false,
        alpha = 1.0
    )
end

function Makie.plot!(p::QuantileHist2D)
    marg_dist = lift(p.samples, p.vsel, p.nbins, p.closed, p.filter) do smpls, vsel, b, c, f
        return MarginalDist(smpls, vsel; bins=b, closed=c, filter=f)
    end

    plot_data = lift(marg_dist, p.normalization, p.levels, p.colormap, p.rev, p.alpha) do marg, norm, levels, cmap, rev, alpha
        d = marg.dist isa BAT.ReshapedDist ? marg.dist.dist : marg.dist
        h_raw = convert(StatsBase.Histogram, d)
        h_norm = norm == :none ? h_raw : StatsBase.normalize(h_raw, mode = norm)

        valid_intervals = sort(filter(x -> 0 < x < 1, levels))
        hists, _ = BAT.get_smallest_intervals(h_norm, valid_intervals)

        pal = cgrad(cmap, length(valid_intervals), categorical=true, rev=!rev, alpha=alpha)
        dims = size(h_norm.weights)
        color_grid = fill(RGBA{Float32}(0,0,0,0), dims)

        for (i, sub_hist) in enumerate(hists)
            color_idx = length(valid_intervals) - i + 1
            c = pal[color_idx]
            mask = sub_hist.weights .> 0
            color_grid[mask] .= c
        end

        centers = BAT.get_bin_centers(marg)
        return (centers[1], centers[2], color_grid)
    end

    x_centers = lift(d -> d[1], plot_data)
    y_centers = lift(d -> d[2], plot_data)
    colors    = lift(d -> d[3], plot_data)

    heatmap!(p, x_centers, y_centers, colors)
    return p
end


@recipe(Hexbin2D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        nbins = (30, 30),
        filter = false,
        colormap = :Blues,
        rev = true,
        alpha = 1.0,
        threshold = nothing
    )
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
            pos_w = w[w .> 0]
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
        weights = w_vals,
        bins = final_bins,
        colormap = final_cmap,
        alpha = p.alpha,
        threshold = final_thresh
    )

    return p
end

