# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Fresh values per call, deliberately not shared consts -- see the
# change-tracking rationale above _empty_scatter2d_primitives
# (makie_scatter.jl).
_empty_kde1d_primitives() = (x=Vector{Float64}(), density=Vector{Float64}(), poly_points=Vector{Point{2,Float32}}())
_empty_kde2d_primitives() = (x=Vector{Float64}(), y=Vector{Float64}(), density=Matrix{Float64}(undef, 0, 0))
_empty_quantilekde1d_primitives() = (polys=Vector{Vector{Point{2,Float32}}}(), fill_colors=Vector{RGBA}(), full_line=Vector{Point{2,Float32}}())
_empty_quantilekde2d_primitives() = (x=Vector{Float64}(), y=Vector{Float64}(), color_grid=Matrix{RGBA{Float32}}(undef, 0, 0))

# KDE2D masks effectively-zero density cells to NaN (rendered transparent,
# matching Hist2D's zero-bin convention) -- as a fraction of the PEAK density,
# not an absolute value: density units are inverse data units, so any absolute
# cutoff is scale-dependent. The previous absolute `> 0.005` blanked the
# entire panel for any parameter whose scale pushes the peak density below it
# (confirmed: sigma ~ 1e4 gives peak ~ 1.6e-9 -- every cell NaN, nothing
# rendered at all, silently), while at unit scale it masked a large chunk of
# genuinely nonzero structure. A relative threshold behaves identically at
# every parameter scale by construction.
const _KDE2D_DENSITY_FLOOR_FRAC = 1e-3

# Weighted counterpart of KernelDensity.default_bandwidth, needed because
# KernelDensity's own bandwidth selection ignores `weights` entirely -- its
# default_bandwidth(data) has no weights-aware method, so both the spread
# estimate and the n^(-1/5) factor come from the raw stored rows. For
# weighted (e.g. Metropolis repeat-count) samples that's systematically
# wrong: confirmed via direct repro that data whose *weighted* distribution
# is a point mass still renders with the full unweighted-spread bandwidth.
#
# Mirrors KernelDensity's robust Silverman rule exactly (same alpha=0.9,
# same min(std, IQR/1.34), same zero-width fallbacks -- see default_bandwidth
# in its univariate.jl) with two substitutions: WEIGHTED std/IQR (uncorrected
# `Weights`, matching this extension's convention everywhere else), and
# n = Kish effective sample size (sum(w)^2 / sum(w^2)) -- which reduces to
# the plain row count for uniform weights, and whose exact choice is
# insensitive anyway under the n^(-1/5) exponent. Returns `nothing` for
# degenerate inputs (zero/non-finite weight sums, n_eff <= 1, non-finite
# width) -- callers then simply fall back to KernelDensity's own default.
function _weighted_kde_bandwidth(values::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, alpha::Float64=0.9)
        sum_w = sum(weights)
        sum_w2 = sum(abs2, weights)
        (isfinite(sum_w) && isfinite(sum_w2) && sum_w > 0 && sum_w2 > 0) || return nothing
        n_eff = sum_w^2 / sum_w2
        n_eff <= 1 && return nothing

        w = Weights(weights)
        var_width = std(values, w)
        q25, q75 = quantile(values, w, [0.25, 0.75])
        quantile_width = (q75 - q25) / 1.34

        width = min(var_width, quantile_width)
        if width == 0.0
                width = var_width == 0.0 ? 1.0 : var_width
        end
        isfinite(width) || return nothing
        return alpha * width * n_eff^(-0.2)
end

# The two kde() entry points all four KDE recipes go through. Uniform
# weights take the fast path with NO explicit bandwidth -- KernelDensity's
# own default is exactly right there (Kish n_eff == row count, and its
# unweighted std/IQR equal the weighted ones), so the common IID/unit-weight
# case stays bit-for-bit identical to calling kde() directly. Only genuinely
# non-uniform weights get the weighted bandwidth; a `nothing` from the
# helper (degenerate data) also falls back to the default.
function _weighted_kde1d(values::AbstractVector, weights::AbstractVector)
        h = allequal(weights) ? nothing : _weighted_kde_bandwidth(values, weights)
        return isnothing(h) ? kde(values; weights=weights) : kde(values; weights=weights, bandwidth=h)
end

# Per-dimension bandwidths, mirroring KernelDensity's own bivariate
# default_bandwidth (which applies the univariate rule per coordinate).
# Falls back entirely (not per-axis) if either dimension is degenerate.
function _weighted_kde2d(x::AbstractVector, y::AbstractVector, weights::AbstractVector)
        hx = hy = nothing
        if !allequal(weights)
                hx = _weighted_kde_bandwidth(x, weights)
                hy = _weighted_kde_bandwidth(y, weights)
        end
        return (isnothing(hx) || isnothing(hy)) ?
                kde((x, y); weights=weights) :
                kde((x, y); weights=weights, bandwidth=(hx, hy))
end

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
        # kde() errors on empty input -- a live cell can still have zero samples
        # (e.g. right after vsel activates, before the first batch flushes, or if
        # buffered samples get cleared later), so degrade like a dead cell instead.
        isempty(weights) && return _empty_kde1d_primitives()
        kde_result = _weighted_kde1d(vec(marg_coords), weights)
        # collect(...): kde_result.x is a StepRangeLen, not a Vector{Float64} --
        # matching _empty_kde1d_primitives()'s declared type here (rather than the
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
        isempty(weights) && return _empty_kde2d_primitives()
        kde_result = _weighted_kde2d(view(marg_coords, 1, :), view(marg_coords, 2, :), weights)
        density = kde_result.density
        # Relative-to-peak cutoff (see _KDE2D_DENSITY_FLOOR_FRAC's comment);
        # single pass instead of the previous fill + mask + masked-assign.
        floor_val = _KDE2D_DENSITY_FLOOR_FRAC * maximum(density)
        nonzero_density = map(d -> d > floor_val ? d : NaN, density)

        # collect(...): see KDE1D's matching comment above -- kde_result.x/.y
        # are StepRangeLen, not Vector{Float64} like _empty_kde2d_primitives().
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


# Contiguous true-runs of `mask`, as index ranges into it. Used by
# QuantileKDE1D below: an HPD region of a multimodal marginal is a union of
# disjoint intervals, and each needs its own filled polygon -- a single
# polygon over the whole mask (the previous implementation) draws its top
# edge straight across every below-threshold valley between modes, filling
# regions that are explicitly NOT part of the credible region (confirmed via
# direct repro on a bimodal marginal: every band spanned the full range,
# valley included).
function _mask_runs(mask::AbstractVector{Bool})
        runs = UnitRange{Int}[]
        start = 0
        for k in eachindex(mask)
                if mask[k]
                        start == 0 && (start = k)
                elseif start != 0
                        push!(runs, start:(k-1))
                        start = 0
                end
        end
        start != 0 && push!(runs, start:lastindex(mask))
        return runs
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
        isempty(weights) && return _empty_quantilekde1d_primitives()
        (; levels) = config
        kde_result = _weighted_kde1d(vec(marg_coords), weights)
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

                # One polygon PER CONTIGUOUS RUN of the mask, not one over the
                # whole mask -- see _mask_runs' comment above. The level's
                # color is pushed once per run so polys/fill_colors stay
                # index-paired (compose passes them together via
                # S.Poly(polys; color=fill_colors)). Unimodal marginals have
                # exactly one run per level, reproducing the old output
                # bit-for-bit there.
                mask = prob_mass .>= threshold
                for run in _mask_runs(mask)
                        x_run = kde_result.x[run]
                        y_run = density[run]
                        pts = Point2f.(
                                vcat(x_run, reverse(x_run)),
                                vcat(y_run, zeros(length(run)))
                        )
                        push!(polys, pts)
                        push!(fill_colors, _quantile_level_color(i))
                end
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
        isempty(weights) && return _empty_quantilekde2d_primitives()
        (; levels) = config
        kde_result = _weighted_kde2d(marg_coords[1, :], marg_coords[2, :], weights)

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
        # are StepRangeLen, not Vector{Float64} like _empty_quantilekde2d_primitives().
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

