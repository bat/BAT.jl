
@recipe(KDE1D, x) do scene
    Attributes(
        color = Makie.wong_colors()[1],
        alpha = 1.0,
        filled = true,
        edge = false,
        strokecolor = Makie.wong_colors()[1],
        strokewidth = 1,
        weights = nothing
    )
end

function Makie.plot!(p::KDE1D)
    kde_result = lift(p[1], p.weights) do x, w
        return kde(x, weights = w)
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


@recipe(KDE2D, x, y) do scene
    Attributes(
        colormap = :Blues,
        alpha = 1.0,
        weights = nothing,
        rev = false
    )
end

function Makie.plot!(p::KDE2D)
    kde_result = lift(p[1], p[2], p.weights) do x, y, w
        return kde((x, y), weights = w)
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


@recipe(QuantileKDE1D, x) do scene
    Attributes(
        weights = nothing,
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
    geometry_data = lift(p[1], p.weights, p.levels, p.colormap, p.alpha, p.rev) do x, w, levels, cmap, alpha, rev
        w_obj = isnothing(w) ? UnitWeights{Float64}(length(x)) : weights(w)
        k = kde(x, weights = w_obj)

        dens = k.density
        prob_mass = dens * step(k.x) 

        sorted_p = sort(prob_mass, rev=true)
        cum_p = cumsum(sorted_p)
        cum_p ./= cum_p[end]

        active_levels = sort(filter(>(0), levels))

        pal = cgrad(cmap, length(active_levels), categorical=true, rev=!rev, alpha=alpha)

        polys = Vector{Point2f}[]
        fill_colors = RGBAf[]

        for (i, level) in enumerate(reverse(active_levels))
            idx = searchsortedfirst(cum_p, level)
            threshold = sorted_p[clamp(idx, 1, length(sorted_p))]

            mask = dens .>= threshold
            x_fill = k.x[mask]
            y_fill = k.density[mask]

            if isempty(x_fill)
                continue
            end

            pts = Point2f.(
                vcat(x_fill, reverse(x_fill)),
                vcat(zeros(length(x_fill)), reverse(y_fill))
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

    full_line = lift(p[1], p.weights) do x, w
        w_obj = isnothing(w) ? UnitWeights{Float64}(length(x)) : weights(w)
        k = kde(x, weights = w_obj)
        return Point2f.(k.x, k.density)
    end

    lines!(p, full_line;
        color = p.strokecolor,
        linewidth = p.strokewidth,
        visible = p.edge
    )

    return p
end


@recipe(QuantileKDE2D, x, y) do scene
    Attributes(
        weights = nothing,
        levels = cdf.(Chi(2), 0:3),
        colormap = :Blues,
        alpha = 1.0,
        rev = false
    )
end


function Makie.plot!(p::QuantileKDE2D)
    plot_data = lift(p[1], p[2], p.weights, p.levels, p.colormap, p.alpha, p.rev) do x, y, w, levels, cmap, alpha, rev
        w_obj = isnothing(w) ? UnitWeights{Float64}(length(x)) : weights(w)
        k = kde((x, y), weights = w_obj)

        Z = k.density
        Z_flat = vec(Z)
        sorted_Z = sort(Z_flat, rev=true)

        cum_p = cumsum(sorted_Z)
        cum_p ./= cum_p[end]

        thresholds = [sorted_Z[searchsortedfirst(cum_p, level)] for level in levels]

        push!(thresholds, 0.0)

        final_levels = reverse(thresholds)
        final_cmap = cgrad(cmap, rev=rev, alpha=alpha)

        return (k.x, k.y, Z, final_levels, final_cmap)
    end

    x_g    = lift(d -> d[1], plot_data)
    y_g    = lift(d -> d[2], plot_data)
    Z      = lift(d -> d[3], plot_data)
    lvls   = lift(d -> d[4], plot_data)
    cmap   = lift(d -> d[5], plot_data)

    contourf!(p, x_g, y_g, Z;
        levels = lvls,
        colormap = cmap
    )

    return p
end

