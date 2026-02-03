
@recipe(KDE1D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        filter = false,
        color = Makie.wong_colors()[1],
        alpha = 1.0,
        filled = true,
        edge = false,
        strokecolor = Makie.wong_colors()[1],
        strokewidth = 1
    )
end

function Makie.plot!(p::KDE1D)
    kde_result = lift(p.samples, p.vsel, p.filter) do smpls, vsel, f
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = drop_low_weight_samples(marg_res)
        end

        vals = flatview(unshaped.(marg_res).v)
        x_data = vec(vals)
        w_data = marg_res.weight

        return kde(x_data, weights = w_data)
    end

    poly_points = lift(kde_result) do k
        vcat(
            Point2f.(k.x, k.density),
            Point2f.(reverse(k.x), 0.0)
        )
    end

    poly!(p, poly_points;
        color = p.color,
        alpha = p.alpha,
        visible = p.filled
    )

    lines!(p, lift(k -> k.x, kde_result), lift(k -> k.density, kde_result);
        color = p.strokecolor,
        linewidth = p.strokewidth,
        visible = p.edge
    )

    return p
end


@recipe(KDE2D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        filter = false,
        colormap = :Blues,
        alpha = 1.0,
        rev = false
    )
end

function Makie.plot!(p::KDE2D)
    kde_result = lift(p.samples, p.vsel, p.filter) do smpls, vsel, f
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = drop_low_weight_samples(marg_res)
        end

        flat_vals = flatview(unshaped.(marg_res).v)
        x_data = flat_vals[1, :]
        y_data = flat_vals[2, :]
        w_data = marg_res.weight

        return kde((x_data, y_data), weights = w_data)
    end

    cmap_final = lift(p.colormap, p.rev) do cm, r
        r ? Reverse(cm) : cm
    end

    heatmap!(p,
        lift(k -> k.x, kde_result),
        lift(k -> k.y, kde_result),
        lift(k -> k.density, kde_result);
        colormap = cmap_final,
        alpha = p.alpha
    )

    return p
end


@recipe(QuantileKDE1D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        filter = false,
        levels = cdf.(Chi(1), 0:3),
        colormap = :Blues,
        alpha = 1.0,
        rev = false,
        edge = false,
        strokecolor = Makie.wong_colors()[1],
        strokewidth = 1
    )
end

function Makie.plot!(p::QuantileKDE1D)
    kde_result = lift(p.samples, p.vsel, p.filter) do smpls, vsel, f
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = drop_low_weight_samples(marg_res)
        end

        vals = flatview(unshaped.(marg_res).v)
        x_data = vec(vals)
        w_data = marg_res.weight

        return kde(x_data, weights = w_data)
    end

    geometry_data = lift(kde_result, p.levels, p.colormap, p.alpha, p.rev) do k, levels, cmap, alpha, rev
        dens = k.density
        step_size = step(k.x)
        prob_mass = dens * step_size 

        sorted_p = sort(prob_mass, rev=true)
        cum_p = cumsum(sorted_p)
        total_p = cum_p[end]

        cum_p ./= total_p

        active_levels = sort(filter(x -> 0 < x < 1, levels))

        pal = cgrad(cmap, length(active_levels), categorical=true, rev=!rev, alpha=alpha)

        polys = Vector{Point2f}[]
        fill_colors = RGBA[]

        for (i, level) in enumerate(reverse(active_levels))
            idx = searchsortedfirst(cum_p, level)
            safe_idx = clamp(idx, 1, length(sorted_p))
            threshold = sorted_p[safe_idx]

            mask = prob_mass .>= threshold
            x_fill = k.x[mask]
            y_fill = k.density[mask]

            if isempty(x_fill)
                continue
            end

            pts = Point2f.(
                vcat(x_fill, reverse(x_fill)),
                vcat(y_fill, zeros(length(x_fill)))
            )
            push!(polys, pts)

            original_idx = length(active_levels) - i + 1
            push!(fill_colors, pal[original_idx])
        end

        return (polys, fill_colors)
    end

    poly_shapes = lift(x -> x[1], geometry_data)
    poly_colors = lift(x -> x[2], geometry_data)

    poly!(p, poly_shapes; color = poly_colors)

    full_line = lift(kde_result) do k
        return Point2f.(k.x, k.density)
    end

    lines!(p, full_line;
        color = p.strokecolor,
        linewidth = p.strokewidth,
        visible = p.edge
    )

    return p
end


@recipe(QuantileKDE2D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        filter = false,
        levels = cdf.(Chi(2), 0:3),
        colormap = :Blues,
        alpha = 1.0,
        rev = false
    )
end

function Makie.plot!(p::QuantileKDE2D)
    kde_result = lift(p.samples, p.vsel, p.filter) do smpls, vsel, f
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = drop_low_weight_samples(marg_res)
        end

        flat_vals = flatview(unshaped.(marg_res).v)
        x_data = flat_vals[1, :]
        y_data = flat_vals[2, :]
        w_data = marg_res.weight

        return kde((x_data, y_data), weights = w_data)
    end

    plot_data = lift(kde_result, p.levels, p.colormap, p.rev, p.alpha) do k, levels, cmap, rev, alpha
        Z = k.density
        Z_flat = vec(Z)
        sorted_Z = sort(Z_flat, rev=true)

        cum_p = cumsum(sorted_Z)
        total_p = cum_p[end]
        cum_p ./= total_p

        thresholds = Float64[]
        valid_levels = sort(filter(x -> 0 < x < 1, levels))

        for level in valid_levels
            idx = searchsortedfirst(cum_p, level)
            safe_idx = clamp(idx, 1, length(sorted_Z))
            push!(thresholds, sorted_Z[safe_idx])
        end

        push!(thresholds, 0.0)
        final_levels = sort(thresholds)

        final_cmap = cgrad(cmap, length(final_levels)-1, categorical=true, rev=rev, alpha=alpha)

        return (k.x, k.y, Z, final_levels, final_cmap)
    end

    x_g  = lift(d -> d[1], plot_data)
    y_g  = lift(d -> d[2], plot_data)
    Z    = lift(d -> d[3], plot_data)
    lvls = lift(d -> d[4], plot_data)
    cmap = lift(d -> d[5], plot_data)

    contourf!(p, x_g, y_g, Z;
        levels = lvls,
        colormap = cmap
    )

    return p
end

