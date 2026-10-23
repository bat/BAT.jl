# This file is a part of BAT.jl, licensed under the MIT License (MIT).

_empty_kde1d_primitives() = (x=Vector{Float64}(), density=Vector{Float64}(), poly_points=Vector{Point{2,Float64}}())
_empty_kde2d_primitives() = (x=Vector{Float64}(), y=Vector{Float64}(), xedges=Vector{Float64}(), yedges=Vector{Float64}(), density=Matrix{Float64}(undef, 0, 0))
_empty_quantilekde1d_primitives() = (polys=Vector{Vector{Point{2,Float64}}}(), fill_colors=Vector{RGBAf}(), full_line=Vector{Point{2,Float64}}())
_empty_quantilekde2d_primitives() = (x=Vector{Float64}(), y=Vector{Float64}(), xedges=Vector{Float64}(), yedges=Vector{Float64}(), color_grid=Matrix{RGBA{Float32}}(undef, 0, 0))

# KDE2D's NaN-mask cutoff, as a fraction of the PEAK density, not an absolute
# value: density units are inverse data units, so any absolute cutoff is scale-dependent.
const _KDE2D_DENSITY_FLOOR_FRAC = 1e-3

function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::KDE1D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_kde1d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::KDE1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    # A selected window can be empty or contain only zero-mass samples.
    # Neither has a density; do not pass zero mass into KDE normalization.
    (isempty(weights) || all(iszero, weights)) && return _empty_kde1d_primitives()
    kde_result = _weighted_kde1d(vec(marg_coords), weights, get(config, :support, nothing))
    # Materialize the grid to match the empty sentinel's Vector{Float64}.
    x = collect(kde_result.x)
    density = kde_result.density
    poly_points = vcat(
        Point{2,Float64}.(x, density),
        Point{2,Float64}.(reverse(x), 0.0)
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

function _kde_check_density_colors(density)
    fail() = throw(ArgumentError(
        "KDE2D visible density exceeds the arithmetic range of Float32 color values; change coordinate units or use QuantileKDE2D"))
    lo, hi = Inf32, -Inf32
    for value in density
        isnan(value) && continue # intentional visibility mask, after raw admission
        converted = Float32(value)
        isfinite(converted) && converted >= floatmin(Float32) || fail()
        lo, hi = min(lo, converted), max(hi, converted)
    end
    isfinite(lo) && isfinite(hi) || fail()
    # Match Makie's automatic range after numeric colors narrow to Float32.
    lo == hi && ((lo, hi) = (lo - 0.5f0, hi + 0.5f0))
    span = hi - lo
    reciprocal = inv(span)
    isfinite(span) && span >= floatmin(Float32) && isfinite(reciprocal) &&
        reciprocal >= 2floatmin(Float32) || fail()
    return nothing
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
    return _empty_kde2d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::KDE2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    (isempty(weights) || all(iszero, weights)) && return _empty_kde2d_primitives()
    kde_result = _weighted_kde2d(view(marg_coords, 1, :), view(marg_coords, 2, :), weights, get(config, :support, nothing))
    density = kde_result.density
    # Relative-to-peak cutoff -- see _KDE2D_DENSITY_FLOOR_FRAC's comment.
    floor_val = _KDE2D_DENSITY_FLOOR_FRAC * maximum(density)
    nonzero_density = map(d -> d > floor_val ? d : NaN, density)

    return (x=collect(kde_result.x), y=collect(kde_result.y),
        xedges=kde_result.xedges, yedges=kde_result.yedges, density=nonzero_density)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::KDE2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; x, y, xedges, yedges, density) = primitives

    if isempty(x)
        return PlotSpec[]
    end

    # Makie's numeric heatmap conversion stores color values as Float32 before
    # applying its color scale. Keep physical density units; reject overflow or
    # lost visible mass instead of silently plotting Inf/zero color values.
    _kde_check_density_colors(density)

    (; rev, colormap) = config
    cmap_final = rev ? Reverse(colormap) : colormap
    heat = transposed ?
        S.Heatmap(yedges, xedges, permutedims(density); colormap=cmap_final) :
        S.Heatmap(xedges, yedges, density; colormap=cmap_final)
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
    return _empty_quantilekde1d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileKDE1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    (isempty(weights) || all(iszero, weights)) && return _empty_quantilekde1d_primitives()
    (; levels) = config
    kde_result = _weighted_kde1d(vec(marg_coords), weights, get(config, :support, nothing))
    x = kde_result.x
    density = kde_result.density

    active_levels = sort(filter(p -> 0 < p < 1, levels))

    polys = Vector{Point{2,Float64}}[]
    # RGBAf (concrete), matching _empty_quantilekde1d_primitives' field type.
    fill_colors = RGBAf[]

    # enumerate(reverse(...)): i=1 is the largest level = loosest/widest region --
    # the ascending-index convention all four _quantile_level_color callers share.
    for (i, level) in enumerate(reverse(active_levels))
        region = _kde_curve_region(x, density, level)
        for component in region.components
            pts = Point{2,Float64}.(
                vcat(component.x, reverse(component.x)),
                vcat(component.y, zeros(length(component.x)))
            )
            push!(polys, pts)
            push!(fill_colors, _quantile_level_color(i))
        end
    end

    full_line = Point{2,Float64}.(x, density)

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
    # Hidden by default; set on this one PlotSpec, not the shared Lines theme --
    # S.Lines is also used by KDE1D's main curve, Cov2D, PDF1D, and Trace2D.
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
    return _empty_quantilekde2d_primitives()
end

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::QuantileKDE2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    (isempty(weights) || all(iszero, weights)) && return _empty_quantilekde2d_primitives()
    (; levels) = config
    kde_result = _weighted_kde2d(marg_coords[1, :], marg_coords[2, :], weights, get(config, :support, nothing))

    density = kde_result.density
    density_flat = vec(density)
    order = sortperm(density_flat; rev=true)
    density_sorted = density_flat[order]
    masses = _kde_cell_mass(density, kde_result.xedges, kde_result.yedges)
    cum_p = cumsum(vec(masses.probabilities)[order])
    cum_p ./= last(cum_p)

    valid_levels = sort(filter(p -> 0 < p < 1, levels))

    # Explicit per-cell colors + Heatmap (like QuantileHist2D), not Contourf --
    # Contourf interpolates band colors at threshold midpoints.
    color_grid = fill(RGBA{Float32}(0, 0, 0, 0), size(density))

    # j=1 is the loosest/widest region (see QuantileKDE1D's matching comment);
    # writing widest-first lets narrower nested regions overwrite their interior.
    for (j, level) in enumerate(reverse(valid_levels))
        idx = searchsortedfirst(cum_p, level)
        safe_idx = clamp(idx, 1, length(density_sorted))
        threshold = density_sorted[safe_idx]
        mask = density .>= threshold
        color_grid[mask] .= _quantile_level_color(j)
    end

    return (x=collect(kde_result.x), y=collect(kde_result.y),
        xedges=kde_result.xedges, yedges=kde_result.yedges, color_grid=color_grid)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::QuantileKDE2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; x, y, xedges, yedges, color_grid) = primitives

    if isempty(x)
        return PlotSpec[]
    end

    # permutedims (not transpose) -- transpose would recurse into the RGBA cells.
    heat = transposed ?
        S.Heatmap(yedges, xedges, permutedims(color_grid)) :
        S.Heatmap(xedges, yedges, color_grid)
    return [heat]
end
