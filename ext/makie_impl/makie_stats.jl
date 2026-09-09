# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Fresh sentinels per call, not shared consts -- see _empty_scatter2d_primitives (makie_scatter.jl).
_empty_cov2d_primitives() = (ellipse_points=Vector{Point2f}(), axes_segments=Vector{Point2f}())
_empty_std1d_primitives() = (positions=Vector{Float64}(),)
_empty_std2d_primitives() = (x_lines=Vector{Float64}(), y_lines=Vector{Float64}())
_empty_mean1d_primitives() = (x=Vector{Float64}(),)
_empty_mean2d_primitives() = (μ_x=Vector{Float64}(), μ_y=Vector{Float64}())
_empty_errorbars1d_primitives() = (μ=Vector{Float64}(), err=Vector{Float64}())
_empty_errorbars2d_primitives() = (μ_x=Vector{Float64}(), μ_y=Vector{Float64}(), err_x=Vector{Float64}(), err_y=Vector{Float64}())
_empty_pdf1d_primitives() = (poly_points=Vector{Point2f}(), x=Float64[], y=Float64[])

# Wrapped in Makie.Linestyle: re-resolving a bare :dash's converted Vector
# pattern (as GLMakie's render loop does) triggers a deprecation warning per frame.
const _MEAN1D_DASH = Makie.Linestyle(Makie.line_pattern(:dash, :normal))

# Shared prologue of the live stats-recipe methods: optionally apply the
# low-weight cutoff (config.filter); `nothing` when no samples survive.
# StatsBase's std/var/cov with ProbabilityWeights default to corrected=false --
# the uncorrected convention this extension uses everywhere.
function _stats_inputs(marg_coords::SubArray, weights::SubArray, config::NamedTuple)
    isempty(weights) && return nothing
    if config.filter
        mask = _low_weight_mask(weights)
        w = view(weights, mask)
        isempty(w) && return nothing
        return view(marg_coords, :, mask), ProbabilityWeights(w)
    end
    return marg_coords, ProbabilityWeights(weights)
end

# Unit circle sampled once, not rebuilt per call.
const _COV_ELLIPSE_CIRCLE = let θ = range(0, 2π, length=200)
    [cos.(θ)'; sin.(θ)']
end

# Shared with the Cov2D dead-cell fallback.
function _cov_ellipse_primitives(μ::AbstractVector, Σ::AbstractMatrix, nsigma::Real)
    vals, vecs = eigen(Σ)
    stds = sqrt.(clamp.(vals, 0, Inf))

    ellipse_mat = vecs * nsigma * Diagonal(stds) * _COV_ELLIPSE_CIRCLE .+ μ
    ellipse_points = Point2f.(ellipse_mat[1, :], ellipse_mat[2, :])

    axes_segments = Point2f[]
    for i in 1:2
        direction = vecs[:, i] * stds[i] * 2 * nsigma
        p1 = Point2f(μ .- direction)
        p2 = Point2f(μ .+ direction)
        push!(axes_segments, p1, p2)
    end
    return (ellipse_points=ellipse_points, axes_segments=axes_segments)
end

function determine_recipe_status(subject::Mean1D, live_recipe::R1) where {R1<:BATMakieRecipe}
    return LiveRecipe()
end

function determine_recipe_status(subject::Std1D, live_recipe::R1) where {R1<:BATMakieRecipe}
    return LiveRecipe()
end

function determine_recipe_status(subject::Mean2D, live_recipe_1::R1, live_recipe_2::R2) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
    return LiveRecipe()
end

function determine_recipe_status(subject::Cov2D, live_recipe_1::R1, live_recipe_2::R2) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
    return LiveRecipe()
end

function determine_recipe_status(subject::Std2D, live_recipe_1::R1, live_recipe_2::R2) where {R1<:BATMakieRecipe,R2<:BATMakieRecipe}
    return LiveRecipe()
end


struct Makie1DStats <: BATMakieRecipe end

function get_stats_plotspecs(
    inputs::NamedTuple,
    vsel::Tuple{Integer,Integer},
    recipe::Makie1DStats,
    config::NamedTuple
)
    i = vsel[1]
    plotspecs = PlotSpec[]

    mean_primitives = getproperty(inputs, primitive_symbol(Mean1D(), (i, i)))
    append!(plotspecs, compose_plotspecs(mean_primitives, Mean1D(), config))

    std_primitives = getproperty(inputs, primitive_symbol(Std1D(), (i, i)))
    append!(plotspecs, compose_plotspecs(std_primitives, Std1D(), config))

    return plotspecs
end

struct Makie2DStats <: BATMakieRecipe end

function get_stats_plotspecs(
    inputs::NamedTuple,
    vsel::Tuple{Integer,Integer},
    recipe::Makie2DStats,
    config::NamedTuple;
    # Forwarded: the 2D stats overlay must follow the cell's main-recipe
    # orientation (see _init_gridlayout's invariant comment).
    transposed::Bool = false
)
    i, j = vsel
    plotspecs = PlotSpec[]

    # Mean2D is deliberately not drawn in the 2D stats overlay; its recipe
    # machinery still computes for live cells.

    cov_primitives = getproperty(inputs, primitive_symbol(Cov2D(), (i, j)))
    append!(plotspecs, compose_plotspecs(cov_primitives, Cov2D(), config; transposed=transposed))

    std_primitives = getproperty(inputs, primitive_symbol(Std2D(), (i, j)))
    append!(plotspecs, compose_plotspecs(std_primitives, Std2D(), config; transposed=transposed))

    return plotspecs
end



function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Cov2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_cov2d_primitives()
end

# Full recompute of cheap weighted moments via _stats_inputs. The isfinite
# guards matter: an all-zero-weight sample set yields NaN moments, and Cov2D's
# eigen() on a non-finite matrix throws.
function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Cov2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_cov2d_primitives()
    coords, w_prob = si
    (; nsigma) = config
    μ = [mean(view(coords, 1, :), w_prob), mean(view(coords, 2, :), w_prob)]
    # Matrix(...): StatsBase.cov(::DenseMatrix, ...) doesn't accept a SubArray.
    Σ = StatsBase.cov(Matrix(coords), w_prob, 2)
    (all(isfinite, μ) && all(isfinite, Σ)) || return _empty_cov2d_primitives()
    return _cov_ellipse_primitives(μ, Σ, nsigma)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Cov2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; ellipse_points, axes_segments) = primitives

    if isempty(ellipse_points)
        return PlotSpec[]
    end

    # Lower-triangle cells swap x/y at compose time (see _init_gridlayout) --
    # point-based primitives swap each point's own coordinates.
    if transposed
        ellipse_points = Point2f[Point2f(p[2], p[1]) for p in ellipse_points]
        axes_segments = Point2f[Point2f(p[2], p[1]) for p in axes_segments]
    end

    # Color set explicitly per PlotSpec, not via the shared theme: Makie's
    # automatic color-cycling (Lines/LineSegments, also VLines/HLines) silently
    # overrides a theme-level color default. The value itself is theme-derived
    # upstream (config.stats_color).
    lines = S.Lines(ellipse_points; color=config.stats_color)
    line_segments = S.LineSegments(axes_segments; color=config.stats_color)

    return [lines, line_segments]
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Std1D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_std1d_primitives()
end

# See Cov2D's matching comment above.
function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Std1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_std1d_primitives()
    coords, w_prob = si
    (; nsigma) = config
    vals = vec(coords)
    μ = mean(vals, w_prob)
    σ = std(vals, w_prob)
    (isfinite(μ) && isfinite(σ)) || return _empty_std1d_primitives()
    return (positions=[μ - nsigma * σ, μ + nsigma * σ],)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Std1D,
    config::NamedTuple
)
    (; positions) = primitives

    if isempty(positions)
        return PlotSpec[]
    end

    # Explicit theme-derived color -- see Cov2D's matching comment above.
    lines = S.VLines(positions; color=config.stats_color)
    return [lines]
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Std2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_std2d_primitives()
end

# See Cov2D's matching comment above.
function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Std2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_std2d_primitives()
    coords, w_prob = si
    (; nsigma) = config
    x, y = view(coords, 1, :), view(coords, 2, :)
    μ_x, μ_y = mean(x, w_prob), mean(y, w_prob)
    σ_x, σ_y = std(x, w_prob), std(y, w_prob)
    all(isfinite, (μ_x, μ_y, σ_x, σ_y)) || return _empty_std2d_primitives()
    return (
        x_lines=[μ_x - nsigma * σ_x, μ_x + nsigma * σ_x],
        y_lines=[μ_y - nsigma * σ_y, μ_y + nsigma * σ_y]
    )
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Std2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; x_lines, y_lines) = primitives

    if isempty(x_lines)
        return PlotSpec[]
    end

    # Transposed cells swap vertical vs horizontal positions -- see Cov2D above.
    transposed && ((x_lines, y_lines) = (y_lines, x_lines))

    # Explicit theme-derived color -- see Cov2D's matching comment above.
    vlines = S.VLines(x_lines; color=config.stats_color)
    hlines = S.HLines(y_lines; color=config.stats_color)
    return [vlines, hlines]
end



function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Mean1D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_mean1d_primitives()
end

# See Cov2D's matching comment above.
function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Mean1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_mean1d_primitives()
    coords, w_prob = si
    μ = mean(vec(coords), w_prob)
    isfinite(μ) || return _empty_mean1d_primitives()
    return (x=[μ],)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Mean1D,
    config::NamedTuple
)
    (; x) = primitives

    if isempty(x)
        return PlotSpec[]
    end

    # _MEAN1D_DASH (see its comment above), set on this PlotSpec only --
    # Std1D/Std2D/Mean2D share the VLines type and must stay solid.
    # Explicit theme-derived color -- see Cov2D's matching comment above.
    lines = S.VLines(x; linestyle=_MEAN1D_DASH, color=config.stats_color)
    return [lines]
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Mean2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_mean2d_primitives()
end

# See Cov2D's matching comment above.
function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Mean2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_mean2d_primitives()
    coords, w_prob = si
    μ_x = mean(view(coords, 1, :), w_prob)
    μ_y = mean(view(coords, 2, :), w_prob)
    (isfinite(μ_x) && isfinite(μ_y)) || return _empty_mean2d_primitives()
    return (μ_x=[μ_x], μ_y=[μ_y])
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Mean2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; μ_x, μ_y) = primitives

    if isempty(μ_x)
        return PlotSpec[]
    end

    # See Std2D's matching comment above.
    transposed && ((μ_x, μ_y) = (μ_y, μ_x))

    # Explicit theme-derived color -- see Cov2D's matching comment above.
    vlines = S.VLines(μ_x; color=config.stats_color)
    hlines = S.HLines(μ_y; color=config.stats_color)
    return [vlines, hlines]
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Errorbars1D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_errorbars1d_primitives()
end

# See Cov2D's matching comment above.
function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Errorbars1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_errorbars1d_primitives()
    coords, w_prob = si
    (; nsigma) = config
    vals = vec(coords)
    μ = mean(vals, w_prob)
    σ = std(vals, w_prob)
    (isfinite(μ) && isfinite(σ)) || return _empty_errorbars1d_primitives()
    return (μ=[μ], err=[σ * nsigma])
end


function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Errorbars1D,
    config::NamedTuple
)
    (; μ, err) = primitives

    if isempty(μ)
        return PlotSpec[]
    end

    (; y_ebars) = config

    ebars = S.Errorbars(μ, [y_ebars], err;
        direction=:x
    )
    points = S.Scatter(μ, [y_ebars])
    return [ebars, points]
end



function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::Errorbars2D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_errorbars2d_primitives()
end

# See Errorbars1D's matching comment above.
function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Errorbars2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_errorbars2d_primitives()
    coords, w_prob = si
    (; nsigma) = config
    x, y = view(coords, 1, :), view(coords, 2, :)
    μ_x, μ_y = mean(x, w_prob), mean(y, w_prob)
    σ_x, σ_y = std(x, w_prob), std(y, w_prob)
    all(isfinite, (μ_x, μ_y, σ_x, σ_y)) || return _empty_errorbars2d_primitives()
    return (μ_x=[μ_x], μ_y=[μ_y], err_x=[σ_x * nsigma], err_y=[σ_y * nsigma])
end


function compose_plotspecs(
    primitives::NamedTuple,
    recipe::Errorbars2D,
    config::NamedTuple;
    transposed::Bool = false
)
    (; μ_x, μ_y, err_x, err_y) = primitives

    if isempty(μ_x)
        return PlotSpec[]
    end

    # See Std2D's matching comment above; the error extents must swap
    # together with their means.
    if transposed
        μ_x, μ_y = μ_y, μ_x
        err_x, err_y = err_y, err_x
    end

    ebars_x = S.Errorbars(μ_x, μ_y, err_x,
        direction=:x
    )
    ebars_y = S.Errorbars(μ_x, μ_y, err_y,
        direction=:y
    )

    point = S.Scatter(μ_x, μ_y)

    return [ebars_x, ebars_y, point]
end


function compute_plotting_primitives(
    ::SubArray,
    ::SubArray,
    ::PDF1D,
    ::RS,
    ::CS,
    ::NamedTuple
) where {RS<:RecipeStatus,CS<:CellStatus}
    return _empty_pdf1d_primitives()
end

# Overlays the Normal distribution fit (by weighted mean/std) to the 1D marginal.
function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::PDF1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_pdf1d_primitives()
    coords, w_prob = si
    (; npoints_pdf) = config

    vals = vec(coords)
    μ = mean(vals, w_prob)
    σ = std(vals, w_prob)
    # A nonempty all-zero-weight vector passes the emptiness guards; std()
    # then returns NaN, which Normal(μ, NaN) rejects with a DomainError.
    (isfinite(μ) && isfinite(σ)) || return _empty_pdf1d_primitives()
    dist = Normal(μ, σ)

    x_min, x_max = σ == 0 ? (μ - 1, μ + 1) : (μ - 4σ, μ + 4σ)
    x = collect(LinRange(x_min, x_max, npoints_pdf))
    y = pdf.(dist, x)

    poly_points = vcat(
        Point2f.(x, y),
        Point2f.(reverse(x), 0.0)
    )

    return (poly_points=poly_points, x=x, y=y)
end

function compose_plotspecs(
    primitives::NamedTuple,
    recipe::PDF1D,
    config::NamedTuple
)
    (; poly_points, x, y) = primitives

    if isempty(x)
        return PlotSpec[]
    end

    (; filled_pdf) = config

    poly = S.Poly(poly_points; visible=filled_pdf)
    lines = S.Lines(x, y)

    return [poly, lines]
end

# See makie_hist.jl's _diag_y_extent for Hist1D for what this is for.
_diag_y_extent(primitives::NamedTuple, ::PDF1D) = isempty(primitives.y) ? 0.0 : maximum(primitives.y)

