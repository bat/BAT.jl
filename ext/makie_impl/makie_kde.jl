# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Dead-cell results are always identical and read-only (isempty-checked, then
# discarded), so these are shared const sentinels instead of fresh allocations
# on every recompute (which happens on every new sample batch for every
# non-selected recipe).
const _EMPTY_KDE1D_PRIMITIVES = (x=Vector{Float64}(), density=Vector{Float64}(), poly_points=Vector{Point{2,Float32}}())
const _EMPTY_KDE2D_PRIMITIVES = (x=Vector{Float64}(), y=Vector{Float64}(), density=Matrix{Float64}(undef, 0, 0))
const _EMPTY_QUANTILEKDE1D_PRIMITIVES = (polys=Vector{Vector{Point{2,Float32}}}(), fill_colors=Vector{RGBA}(), full_line=Vector{Point{2,Float32}}())
const _EMPTY_QUANTILEKDE2D_PRIMITIVES = (x=Vector{Float64}(), y=Vector{Float64}(), color_grid=Matrix{RGBA{Float32}}(undef, 0, 0))

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
        # kde() errors on empty input -- a live cell can still have zero samples
        # (e.g. right after vsel activates, before the first batch flushes, or if
        # buffered samples get cleared later), so degrade like a dead cell instead.
        isempty(weights) && return _EMPTY_KDE1D_PRIMITIVES
        kde_result = kde(vec(marg_coords), weights=weights)
        # collect(...): kde_result.x is a StepRangeLen, not a Vector{Float64} --
        # matching _EMPTY_KDE1D_PRIMITIVES's declared type here (rather than the
        # other way around) avoids the same live/dead ComputePipeline TypedEdge
        # type-lock crash documented for ChainScatter2D/Scatter2D/Hexbin2D
        # (a live cell resolving first locks in StepRangeLen; a later live->dead
        # transition then fails to convert the dead branch's Vector{Float64}
        # into that locked type). Confirmed via direct reproduction: cycling the
        # diagonal recipe then deselecting a variable crashed with exactly
        # "Cannot convert Vector{Float64} to StepRangeLen{...}".
        x = collect(kde_result.x)
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

# See makie_hist.jl's _diag_y_extent for Hist1D for what this is for.
_diag_y_extent(primitives::NamedTuple, ::KDE1D) = isempty(primitives.density) ? 0.0 : maximum(primitives.density)


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
        isempty(weights) && return _EMPTY_KDE2D_PRIMITIVES
        kde_result = kde(marg_coords', weights=weights)
        density = kde_result.density
        nonzero_density = fill(NaN, size(density))
        nonzero_idxs = density .> 0.005
        nonzero_density[nonzero_idxs] .= density[nonzero_idxs]

        # collect(...): see KDE1D's matching comment above -- kde_result.x/.y
        # are StepRangeLen, not Vector{Float64} like _EMPTY_KDE2D_PRIMITIVES.
        return (x=collect(kde_result.x), y=collect(kde_result.y), density=nonzero_density)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::KDE2D,
        config::NamedTuple;
        transposed::Bool=false
)
        (; x, y, density) = primitives

        if isempty(x)
                return PlotSpec[]
        end

        (; rev, colormap) = config
        cmap_final = rev ? Reverse(colormap) : colormap
        # permutedims (not transpose) -- see Hist2D's matching comment
        # (makie_hist.jl) for why, shared with all four heatmap-based recipes.
        heat = transposed ?
                S.Heatmap(y, x, permutedims(density); colormap=cmap_final) :
                S.Heatmap(x, y, density; colormap=cmap_final)
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
        isempty(weights) && return _EMPTY_QUANTILEKDE1D_PRIMITIVES
        (; levels) = config
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

        polys = Vector{Point2f}[]
        fill_colors = RGBA[]

        # enumerate(reverse(active_levels)): i=1 is the *largest* level value
        # (e.g. 0.9973), which needs the lowest density threshold to reach --
        # confirmed directly that this makes i=1 the loosest/widest region and
        # the last i the tightest/narrowest (closest to the peak), the same
        # ascending-index convention _quantile_level_color's other three
        # callers (QuantileHist1D/2D, QuantileKDE2D) already use.
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

                push!(fill_colors, _quantile_level_color(i))
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
        # visible=false by default, per explicit request -- this outline
        # traces the full (unclipped) KDE curve on top of the filled
        # credible-region bands, which visually reads as an odd extra edge
        # drawn over the plot. Set directly on this one PlotSpec (not via
        # the shared `Lines` theme block, the way Hist1D's Stairs outline
        # was hidden) since S.Lines is also used by KDE1D's own main curve,
        # Cov2D's ellipse, PDF1D, and Trace2D -- a theme-level default would
        # have hidden all of those too.
        lines = S.Lines(full_line; visible=false)

        return [polyspec, lines]
end

_diag_y_extent(primitives::NamedTuple, ::QuantileKDE1D) = isempty(primitives.full_line) ? 0.0 : maximum(p -> p[2], primitives.full_line)



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
        isempty(weights) && return _EMPTY_QUANTILEKDE2D_PRIMITIVES
        (; levels) = config
        kde_result = kde((marg_coords[1, :], marg_coords[2, :]), weights=weights)

        density = kde_result.density
        density_flat = vec(density)
        density_sorted = sort(density_flat, rev=true)

        cum_p = cumsum(density_sorted)
        total_p = cum_p[end]
        cum_p ./= total_p

        valid_levels = sort(filter(x -> 0 < x < 1, levels))

        # Explicit per-cell color grid + Heatmap, instead of Contourf(levels=,
        # colormap=...) -- mirrors QuantileHist2D's own color_grid+Heatmap
        # approach exactly (makie_hist.jl), and for the same reason:
        # confirmed directly that Contourf's built-in colormap sampling
        # evaluates each *band*'s color at the arithmetic midpoint of that
        # band's threshold range (Makie's own calculate_contourf_polys!,
        # levelcenters = (highs.+lows)./2), mapped through a continuous (not
        # flat/stepped) gradient -- for the highly non-uniform threshold
        # spacing real KDE density levels typically have (can span orders of
        # magnitude), that midpoint essentially never lands exactly on one of
        # the intended color "stops", so a band renders as a visible blend of
        # two neighboring colors instead of a clean, flat one. Reported
        # directly: the middle band showed as a dirty yellow-green blend
        # instead of clear yellow. An explicit per-cell color sidesteps
        # colormap interpolation entirely -- Heatmap just draws each cell's
        # own final RGBA directly, no continuous sampling involved. Bonus:
        # this also removes the previous Contourf-specific need for an
        # "extra unused" level/color slot (density_sorted[1]'s unconditional
        # push) -- no `levels` list is being handed to Makie anymore, so
        # there's nothing for it to reject as degenerate.
        color_grid = fill(RGBA{Float32}(0, 0, 0, 0), size(density))

        # enumerate(reverse(valid_levels)): j=1 is the largest level value
        # (e.g. 0.9973), which needs the lowest density threshold to reach --
        # confirmed directly that this makes j=1 the loosest/widest region
        # (the same ascending-index convention _quantile_level_color's other
        # callers use -- see QuantileKDE1D's matching comment). Writing
        # color_grid in this order (widest region's color first, then
        # progressively narrower -- and therefore strictly nested-subset --
        # regions overwriting their own inner portion) is what produces
        # correctly nested rings, the same "later write wins" mechanism
        # QuantileHist2D's own color_grid construction relies on.
        for (j, level) in enumerate(reverse(valid_levels))
                idx = searchsortedfirst(cum_p, level)
                safe_idx = clamp(idx, 1, length(density_sorted))
                threshold = density_sorted[safe_idx]
                mask = density .>= threshold
                color_grid[mask] .= _quantile_level_color(j)
        end

        # collect(...): see KDE1D's matching comment above -- kde_result.x/.y
        # are StepRangeLen, not Vector{Float64} like _EMPTY_QUANTILEKDE2D_PRIMITIVES.
        return (x=collect(kde_result.x), y=collect(kde_result.y), color_grid=color_grid)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::QuantileKDE2D,
        config::NamedTuple;
        transposed::Bool=false
)
        (; x, y, color_grid) = primitives

        if isempty(x)
                return PlotSpec[]
        end

        # permutedims (not transpose) -- see Hist2D's matching comment
        # (makie_hist.jl); transpose would recurse into the RGBA cells here.
        heat = transposed ?
                S.Heatmap(y, x, permutedims(color_grid)) :
                S.Heatmap(x, y, color_grid)
        return [heat]
end

