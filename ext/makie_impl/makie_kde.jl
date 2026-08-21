# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Fresh sentinels per call, not shared consts -- see _empty_scatter2d_primitives (makie_scatter.jl).
_empty_kde1d_primitives() = (x=Vector{Float64}(), density=Vector{Float64}(), poly_points=Vector{Point{2,Float32}}())
_empty_kde2d_primitives() = (x=Vector{Float64}(), y=Vector{Float64}(), density=Matrix{Float64}(undef, 0, 0))
_empty_quantilekde1d_primitives() = (polys=Vector{Vector{Point{2,Float32}}}(), fill_colors=Vector{RGBAf}(), full_line=Vector{Point{2,Float32}}())
_empty_quantilekde2d_primitives() = (x=Vector{Float64}(), y=Vector{Float64}(), color_grid=Matrix{RGBA{Float32}}(undef, 0, 0))

# KDE2D's NaN-mask cutoff, as a fraction of the PEAK density, not an absolute
# value: density units are inverse data units, so any absolute cutoff is scale-dependent.
const _KDE2D_DENSITY_FLOOR_FRAC = 1e-3

# Weighted counterpart of KernelDensity.default_bandwidth, whose own selection
# ignores `weights` entirely: the same robust Silverman rule with WEIGHTED std/IQR
# (uncorrected Weights, the extension-wide convention) and n = Kish effective
# sample size. Returns `nothing` for degenerate inputs (callers fall back to the default).
function _weighted_kde_bandwidth(values::AbstractVector{<:Real}, weights::AbstractVector{<:Real}, alpha::Float64 = 0.9)
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

# KernelDensity's own FFT-safety padding factor, reused as both the grid padding
# and the reflection cutoff (kernels further than 4h from a bound contribute nothing there).
const _KDE_CUTOFF_BANDWIDTHS = 4.0

# Whether a (lo, hi) support entry actually constrains anything: at least
# one finite bound, and non-degenerate (lo < hi).
_support_is_binding(lo::Real, hi::Real) = (isfinite(lo) || isfinite(hi)) && lo < hi

# Bandwidth when boundary reflection is active: weighted rule for non-uniform
# weights, KernelDensity's default otherwise -- always from the ORIGINAL
# (clamped) data, never the reflection-augmented set.
function _kde_bandwidth(values::AbstractVector, weights::AbstractVector)
    h = allequal(weights) ? nothing : _weighted_kde_bandwidth(values, weights)
    return isnothing(h) ? KernelDensity.default_bandwidth(values) : h
end

# kde() entry point for all KDE recipes; returns a plain (x, density) NamedTuple.
# Uniform weights without support bounds stay bit-identical to plain kde().
# Boundary reflection at finite bounds: clamp float-error leakage, bandwidth from
# the ORIGINAL (clamped) data, mirror samples within 4h of each finite bound
# (the grid must COVER the mirrors -- tabulate() silently drops out-of-grid
# mass), then truncate to the support and renormalize.
function _weighted_kde1d(values::AbstractVector, weights::AbstractVector, support = nothing)
    lo, hi = isnothing(support) ? (-Inf, Inf) : (Float64(support[1]), Float64(support[2]))
    if !_support_is_binding(lo, hi)
        h = allequal(weights) ? nothing : _weighted_kde_bandwidth(values, weights)
        k = isnothing(h) ? kde(values; weights=weights) : kde(values; weights=weights, bandwidth=h)
        return (x=k.x, density=k.density)
    end

    vals = clamp.(Float64.(values), lo, hi)
    w = Float64.(weights)
    h = _kde_bandwidth(vals, w)
    cut = _KDE_CUTOFF_BANDWIDTHS * h

    aug_v = copy(vals)
    aug_w = copy(w)
    for (v, wt) in zip(vals, w)
        if isfinite(lo) && v < lo + cut
            push!(aug_v, 2.0 * lo - v)
            push!(aug_w, wt)
        end
        if isfinite(hi) && v > hi - cut
            push!(aug_v, 2.0 * hi - v)
            push!(aug_w, wt)
        end
    end

    glo = isfinite(lo) ? lo - cut : minimum(vals) - cut
    ghi = isfinite(hi) ? hi + cut : maximum(vals) + cut
    k = kde(aug_v; weights=aug_w, bandwidth=h, boundary=(glo, ghi))

    # Range-indexing keeps k.x a range, so downstream step(x) still works (QuantileKDE1D).
    i1 = searchsortedfirst(k.x, lo)
    i2 = searchsortedlast(k.x, hi)
    i1 > i2 && return (x=k.x, density=k.density)
    xr = k.x[i1:i2]
    dens = k.density[i1:i2]
    total = sum(dens) * step(k.x)
    total > 0 && (dens ./= total)
    return (x=xr, density=dens)
end

# Per-dimension bandwidths, mirroring KernelDensity's bivariate default.
# Boundary reflection as in the 1D case above, per dimension, plus corner
# mirrors for points near two finite bounds at once.
function _weighted_kde2d(x::AbstractVector, y::AbstractVector, weights::AbstractVector, support = nothing)
    (xlo, xhi), (ylo, yhi) = isnothing(support) ? ((-Inf, Inf), (-Inf, Inf)) :
        ((Float64(support[1][1]), Float64(support[1][2])), (Float64(support[2][1]), Float64(support[2][2])))
    if !(_support_is_binding(xlo, xhi) || _support_is_binding(ylo, yhi))
        hx = hy = nothing
        if !allequal(weights)
            hx = _weighted_kde_bandwidth(x, weights)
            hy = _weighted_kde_bandwidth(y, weights)
        end
        k = (isnothing(hx) || isnothing(hy)) ?
            kde((x, y); weights=weights) :
            kde((x, y); weights=weights, bandwidth=(hx, hy))
        return (x=k.x, y=k.y, density=k.density)
    end

    _support_is_binding(xlo, xhi) || ((xlo, xhi) = (-Inf, Inf))
    _support_is_binding(ylo, yhi) || ((ylo, yhi) = (-Inf, Inf))

    xs = clamp.(Float64.(x), xlo, xhi)
    ys = clamp.(Float64.(y), ylo, yhi)
    w = Float64.(weights)
    hx = _kde_bandwidth(xs, w)
    hy = _kde_bandwidth(ys, w)
    cutx = _KDE_CUTOFF_BANDWIDTHS * hx
    cuty = _KDE_CUTOFF_BANDWIDTHS * hy

    ax = copy(xs)
    ay = copy(ys)
    aw = copy(w)
    xrefl = Float64[]
    yrefl = Float64[]
    for i in eachindex(xs)
        empty!(xrefl)
        empty!(yrefl)
        isfinite(xlo) && xs[i] < xlo + cutx && push!(xrefl, 2.0 * xlo - xs[i])
        isfinite(xhi) && xs[i] > xhi - cutx && push!(xrefl, 2.0 * xhi - xs[i])
        isfinite(ylo) && ys[i] < ylo + cuty && push!(yrefl, 2.0 * ylo - ys[i])
        isfinite(yhi) && ys[i] > yhi - cuty && push!(yrefl, 2.0 * yhi - ys[i])
        for xr in xrefl
            push!(ax, xr); push!(ay, ys[i]); push!(aw, w[i])
        end
        for yr in yrefl
            push!(ax, xs[i]); push!(ay, yr); push!(aw, w[i])
        end
        for xr in xrefl, yr in yrefl
            push!(ax, xr); push!(ay, yr); push!(aw, w[i])
        end
    end

    gxlo = isfinite(xlo) ? xlo - cutx : minimum(xs) - cutx
    gxhi = isfinite(xhi) ? xhi + cutx : maximum(xs) + cutx
    gylo = isfinite(ylo) ? ylo - cuty : minimum(ys) - cuty
    gyhi = isfinite(yhi) ? yhi + cuty : maximum(ys) + cuty
    k = kde((ax, ay); weights=aw, bandwidth=(hx, hy), boundary=((gxlo, gxhi), (gylo, gyhi)))

    i1 = searchsortedfirst(k.x, xlo)
    i2 = searchsortedlast(k.x, xhi)
    j1 = searchsortedfirst(k.y, ylo)
    j2 = searchsortedlast(k.y, yhi)
    (i1 > i2 || j1 > j2) && return (x=k.x, y=k.y, density=k.density)
    xr = k.x[i1:i2]
    yr = k.y[j1:j2]
    dens = k.density[i1:i2, j1:j2]
    total = sum(dens) * step(k.x) * step(k.y)
    total > 0 && (dens ./= total)
    return (x=xr, y=yr, density=dens)
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
    # kde() errors on empty input -- a live cell can still have zero samples.
    isempty(weights) && return _empty_kde1d_primitives()
    kde_result = _weighted_kde1d(vec(marg_coords), weights, get(config, :support, nothing))
    # collect(...): kde_result.x is a StepRangeLen; matching the empty sentinel's
    # Vector{Float64} avoids the live/dead ComputePipeline TypedEdge type-lock crash.
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
    kde_result = _weighted_kde2d(view(marg_coords, 1, :), view(marg_coords, 2, :), weights, get(config, :support, nothing))
    density = kde_result.density
    # Relative-to-peak cutoff -- see _KDE2D_DENSITY_FLOOR_FRAC's comment.
    floor_val = _KDE2D_DENSITY_FLOOR_FRAC * maximum(density)
    nonzero_density = map(d -> d > floor_val ? d : NaN, density)

    # collect(...): see KDE1D's matching comment above.
    return (x=collect(kde_result.x), y=collect(kde_result.y), density=nonzero_density)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::KDE2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; x, y, density) = primitives

    if isempty(x)
        return PlotSpec[]
    end

    (; rev, colormap) = config
    cmap_final = rev ? Reverse(colormap) : colormap
    # permutedims (not transpose) -- see Hist2D's matching comment (makie_hist.jl).
    heat = transposed ?
        S.Heatmap(y, x, permutedims(density); colormap=cmap_final) :
        S.Heatmap(x, y, density; colormap=cmap_final)
    return [heat]
end


# Contiguous true-runs of `mask`, as index ranges. Each run gets its own filled
# polygon: a multimodal HPD region must not bridge below-threshold valleys.
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
    kde_result = _weighted_kde1d(vec(marg_coords), weights, get(config, :support, nothing))
    x = kde_result.x
    density = kde_result.density

    step_size = step(x)
    prob_mass = density * step_size

    sorted_p = sort(prob_mass, rev=true)
    cum_p = cumsum(sorted_p)
    total_p = cum_p[end]

    cum_p ./= total_p

    # `p ->` (not `x ->`): the lambda variable would shadow the KDE grid `x` above.
    active_levels = sort(filter(p -> 0 < p < 1, levels))

    polys = Vector{Point2f}[]
    # RGBAf (concrete), matching _empty_quantilekde1d_primitives' field type.
    fill_colors = RGBAf[]

    # enumerate(reverse(...)): i=1 is the largest level = loosest/widest region --
    # the ascending-index convention all four _quantile_level_color callers share.
    for (i, level) in enumerate(reverse(active_levels))
        idx = searchsortedfirst(cum_p, level)
        safe_idx = clamp(idx, 1, length(sorted_p))
        threshold = sorted_p[safe_idx]

        # One polygon per contiguous run -- see _mask_runs above; the color is
        # pushed once per run so polys/fill_colors stay index-paired.
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
    isempty(weights) && return _empty_quantilekde2d_primitives()
    (; levels) = config
    kde_result = _weighted_kde2d(marg_coords[1, :], marg_coords[2, :], weights, get(config, :support, nothing))

    density = kde_result.density
    density_flat = vec(density)
    density_sorted = sort(density_flat, rev=true)

    cum_p = cumsum(density_sorted)
    total_p = cum_p[end]
    cum_p ./= total_p

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

    # collect(...): see KDE1D's matching comment above.
    return (x=collect(kde_result.x), y=collect(kde_result.y), color_grid=color_grid)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::QuantileKDE2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; x, y, color_grid) = primitives

    if isempty(x)
        return PlotSpec[]
    end

    # permutedims (not transpose) -- transpose would recurse into the RGBA cells.
    heat = transposed ?
        S.Heatmap(y, x, permutedims(color_grid)) :
        S.Heatmap(x, y, color_grid)
    return [heat]
end

