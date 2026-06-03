
function compute_plotting_primitives(
        ::Nothing,
        ::Nothing,
        ::KDE1D,
        ::NamedTuple
)
        return (
                x=StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64}(),
                density=Vector{Float64}(),
                poly_points=Vector{Point{2,Float32}}()
        )
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::KDE1D,
        config::NamedTuple
)
        kde_result = kde(vec(marg_coords), weights=weights)
        x = kde_result.x
        density = kde_result.density
        poly_points = vcat(
                Point2f.(x, density),
                Point2f.(reverse(x), 0.0)
        )
        return (x=x, density, poly_points=poly_points)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::KDE1D,
        config::NamedTuple
)
        (; x, density, poly_points) = primitives
        (; color, alpha, filled, strokecolor, strokewidth, edge) = config
        polys = S.Poly(poly_points;
                color=color,
                alpha=alpha,
                visible=filled
        )

        lines = S.Lines(x, density;
                color=strokecolor,
                linewidth=strokewidth,
                visible=edge
        )
        return [polys, lines]
end


function compute_plotting_primitives(
        ::Nothing,
        ::Nothing,
        ::KDE2D,
        ::NamedTuple
)
        return (
                x=StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64}(),
                y=StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64}(),
                density=Matrix{Float64}
        )
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::KDE2D,
        config::NamedTuple
)
        kde_result = kde(marg_coords', weights=weights)
        return (x=kde_result.x, y=kde_result.y, density=kde_result.density)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::KDE2D,
        config::NamedTuple
)
        (; x, y, density) = primitives
        (; alpha, rev, colormap) = config
        cmap_final = rev ? Reverse(colormap) : colormap
        heat = S.Heatmap(x, y, density;
                colormap=cmap_final,
                alpha=alpha
        )
        return [heat]
end


function compute_plotting_primitives(
        ::Nothing,
        ::Nothing,
        ::QuantileKDE1D,
        ::NamedTuple
)
        return (
                polys=Vector{Vector{Point{2,Float32}}}(),
                fill_colors=Vector{RGBA}(),
                full_line=Vector{Point{2,Float32}}()
        )
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::QuantileKDE1D,
        config::NamedTuple
)
        (; levels, colormap, alpha, rev) = config
        kde_result = kde(vec(marg_coords), weights=weights)
        x = kde_result.x
        density = kde_result.density

        step_size = step(x)
        prob_mass = density * step_size

        sorted_p = sort(prob_mass, rev=true)
        cum_p = cumsum(sorted_p)
        total_p = cum_p[end]

        cum_p ./= total_p

        active_levels = sort(filter(x -> 0 < x < 1, levels))

        pal = cgrad(colormap, length(active_levels), categorical=true, rev=!rev, alpha=alpha)

        polys = Vector{Point2f}[]
        fill_colors = RGBA[]

        for (i, level) in enumerate(reverse(active_levels))
                idx = searchsortedfirst(cum_p, level)
                safe_idx = clamp(idx, 1, length(sorted_p))
                threshold = sorted_p[safe_idx]

                mask = prob_mass .>= threshold
                x_fill = kde_result.x[mask]
                y_fill = density[mask]

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


        full_line = Point2f.(x, density)

        return (polys=polys, fill_colors=fill_colors, full_line=full_line)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::QuantileKDE1D,
        config::NamedTuple
)
        (; polys, fill_colors, full_line) = primitives
        (; strokecolor, strokewidth, edge) = config

        polyspec = S.Poly(polys; color=fill_colors)

        lines = S.Lines(full_line;
                color=strokecolor,
                linewidth=strokewidth,
                visible=edge
        )

        return [polyspec, lines]
end



function compute_plotting_primitives(
        ::Nothing,
        ::Nothing,
        ::QuantileKDE2D,
        ::NamedTuple
)
        return (
                x=StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64}(),
                y=StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64}(),
                density=Matrix{Float64}
        )
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::QuantileKDE2D,
        config::NamedTuple
)
        (; levels) = config
        kde_result = kde((marg_coords[1, :], marg_coords[2, :]), weights=weights)

        density = kde_result.density
        density_flat = vec(density)
        density_sorted = sort(density_flat, rev=true)

        cum_p = cumsum(density_sorted)
        total_p = cum_p[end]
        cum_p ./= total_p

        thresholds = Float64[]
        valid_levels = sort(filter(x -> 0 < x < 1, levels))

        for level in valid_levels
                idx = searchsortedfirst(cum_p, level)
                safe_idx = clamp(idx, 1, length(density_sorted))
                push!(thresholds, density_sorted[safe_idx])
        end

        push!(thresholds, 0.0)
        final_levels = sort(thresholds)

        return (x=kde_result.x, y=kde_result.y, density=density, final_levels=final_levels)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::QuantileKDE2D,
        config::NamedTuple
)
        (; x, y, density, final_levels) = primitives
        (; rev, alpha, colormap) = config
        final_cmap = cgrad(colormap, length(final_levels) - 1, categorical=true, rev=rev, alpha=alpha)

        contour = S.Contourf(x, y, density;
                levels=final_levels,
                colormap=final_cmap
        )
        return [contour]
end

