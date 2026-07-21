# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Dead-cell results are always identical and read-only (isempty-checked, then
# discarded), so these are shared const sentinels instead of fresh allocations
# on every recompute (which happens on every new sample batch for every
# non-selected recipe).
const _EMPTY_KDE1D_PRIMITIVES = (x=Vector{Float64}(), density=Vector{Float64}(), poly_points=Vector{Point{2,Float32}}())
const _EMPTY_KDE2D_PRIMITIVES = (x=Vector{Float64}(), y=Vector{Float64}(), density=Matrix{Float64}(undef, 0, 0))
const _EMPTY_QUANTILEKDE1D_PRIMITIVES = (polys=Vector{Vector{Point{2,Float32}}}(), fill_colors=Vector{RGBA}(), full_line=Vector{Point{2,Float32}}())
const _EMPTY_QUANTILEKDE2D_PRIMITIVES = (x=Vector{Float64}(), y=Vector{Float64}(), density=Matrix{Float64}(undef, 0, 0), final_levels=Vector{Float64}(), colors=Vector{RGBA{Float64}}())

function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::KDE1D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return _EMPTY_KDE1D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::KDE1D,
        ::LiveRecipe,
        ::LiveCell,
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

        if isempty(x)
                return PlotSpec[]
        end

        polys = S.Poly(poly_points)
        lines = S.Lines(x, density)
        return [polys, lines]
end


function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::KDE2D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return _EMPTY_KDE2D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::KDE2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        kde_result = kde(marg_coords', weights=weights)
        density = kde_result.density
        nonzero_density = fill(NaN, size(density))
        nonzero_idxs = density .> 0.005
        nonzero_density[nonzero_idxs] .= density[nonzero_idxs]

        return (x=kde_result.x, y=kde_result.y, density=nonzero_density)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::KDE2D,
        config::NamedTuple
)
        (; x, y, density) = primitives

        if isempty(x)
                return PlotSpec[]
        end

        (; rev, colormap) = config
        cmap_final = rev ? Reverse(colormap) : colormap
        heat = S.Heatmap(x, y, density;
                colormap=cmap_final
        )
        return [heat]
end


function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::QuantileKDE1D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return _EMPTY_QUANTILEKDE1D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::QuantileKDE1D,
        ::LiveRecipe,
        ::LiveCell,
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

        pal = cgrad(colormap, rev=rev, alpha=alpha)
        pal_values = collect(range(0.05, 0.7, length(active_levels)))

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

                push!(fill_colors, pal[pal_values[i]])
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

        if isempty(full_line)
                return PlotSpec[]
        end

        polyspec = S.Poly(polys; color=fill_colors)
        lines = S.Lines(full_line)

        return [polyspec, lines]
end



function compute_plotting_primitives(
        ::SubArray,
        ::SubArray,
        ::QuantileKDE2D,
        ::RS,
        ::CS,
        ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
        return _EMPTY_QUANTILEKDE2D_PRIMITIVES
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::QuantileKDE2D,
        ::LiveRecipe,
        ::LiveCell,
        config::NamedTuple
)
        (; levels, rev, colormap, alpha) = config
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

        push!(thresholds, density_sorted[1])
        final_levels = sort(thresholds)

        pal = cgrad(colormap, rev=rev, alpha=alpha)
        pal_values = collect(range(0.05, 0.7, length(final_levels)))
        colors = pal[pal_values]

        return (x=kde_result.x, y=kde_result.y, density=density, final_levels=final_levels, colors=colors)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::QuantileKDE2D,
        config::NamedTuple
)
        (; x, y, density, final_levels, colors) = primitives

        if isempty(x)
                return PlotSpec[]
        end

        contour = S.Contourf(x, y, density;
                levels=final_levels,
                colormap=colors
        )
        return [contour]
end

