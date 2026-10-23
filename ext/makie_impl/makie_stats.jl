# This file is a part of BAT.jl, licensed under the MIT License (MIT).

_empty_cov2d_primitives() = (ellipse_points=Vector{Point{2,Float64}}(), axes_segments=Vector{Point{2,Float64}}())
_empty_std1d_primitives() = (positions=Vector{Float64}(),)
_empty_std2d_primitives() = (x_lines=Vector{Float64}(), y_lines=Vector{Float64}())
_empty_mean1d_primitives() = (x=Vector{Float64}(),)
_empty_mean2d_primitives() = (μ_x=Vector{Float64}(), μ_y=Vector{Float64}())
_empty_errorbars1d_primitives() = (μ=Vector{Float64}(), err=Vector{Float64}())
_empty_errorbars2d_primitives() = (μ_x=Vector{Float64}(), μ_y=Vector{Float64}(), err_x=Vector{Float64}(), err_y=Vector{Float64}())
_empty_pdf1d_primitives() = (poly_points=Vector{Point{2,Float64}}(), x=Float64[], y=Float64[])

# Wrapped in Makie.Linestyle: re-resolving a bare :dash's converted Vector
# pattern (as GLMakie's render loop does) triggers a deprecation warning per frame.
const _MEAN1D_DASH = let pattern = Makie.line_pattern(:dash, :normal)
    # Makie 0.24.16 already returns Linestyle; earlier releases return a vector.
    pattern isa Makie.Linestyle ? pattern : Makie.Linestyle(pattern)
end

# Mean, standard-deviation and covariance specs set the theme-derived
# config.stats_color explicitly: Makie cycling can override a theme default.

# Relative, owned weights for population moments. Validate original positive
# coordinates before filtering; zero-mass rows never enter weighted products.
function _stats_inputs(marg_coords::SubArray, weights::SubArray, config::NamedTuple)
    BAT._plot_validate_weights(weights)
    BAT._plot_validate_coordinates(Tuple(eachrow(marg_coords)), weights)
    isempty(weights) && return nothing
    keep = weights .> 0
    config.filter && (keep .&= _low_weight_mask(weights))
    any(keep) || return nothing
    w = BAT._plot_relative_weights(view(weights, keep); unit_sum=true)
    return view(marg_coords, :, keep), ProbabilityWeights(w)
end

function _stats_mean(values, weights)
    μ = allequal(values) ? first(values) : mean(values, weights)
    isfinite(μ) || BAT._plot_weight_error("weighted mean is outside the working numerical range")
    return μ
end

function _stats_std(values, weights)
    allequal(values) && return zero(float(first(values)))
    σ = std(values, weights; mean=_stats_mean(values, weights), corrected=false)
    isfinite(σ) && σ > 0 || BAT._plot_weight_error("nonconstant weighted spread is outside the working numerical range")
    return σ
end

function _stats_validate_nsigma(nsigma)
    isfinite(nsigma) && nsigma >= 0 || throw(ArgumentError("nsigma must be finite and nonnegative"))
    return nothing
end

function _stats_extent(σ, nsigma)
    _stats_validate_nsigma(nsigma)
    extent = σ * nsigma
    isfinite(extent) && (iszero(σ) || iszero(nsigma) || extent > 0) ||
        BAT._plot_weight_error("statistical extent is outside the working numerical range")
    return BAT._plot_float64(extent, "statistical extent")
end

function _stats_limits(μ, σ, nsigma)
    extent = _stats_extent(σ, nsigma)
    lo = BAT._plot_float64(μ - extent, "statistical lower endpoint")
    hi = BAT._plot_float64(μ + extent, "statistical upper endpoint")
    (iszero(extent) || lo < hi) || BAT._plot_weight_error("statistical endpoints are indistinguishable as Float64")
    return [lo, hi]
end

_stats_position(μ) = BAT._plot_float64(μ, "statistical position")

# Unit circle sampled once, not rebuilt per call.
const _COV_ELLIPSE_CIRCLE = let θ = range(0, 2π, length=200)
    [cos.(θ)'; sin.(θ)']
end

# Ellipse samples and principal-axis endpoints for one covariance, both at the
# SAME nsigma radius: `direction` is a half-length applied in either direction,
# so each segment spans its full axis and its endpoints land on the ellipse.
function _cov_ellipse_primitives(μ::AbstractVector, Σ::AbstractMatrix, nsigma::Real)
    isfinite(nsigma) && nsigma >= 0 || throw(ArgumentError("nsigma must be finite and nonnegative"))
    vals, vecs = eigen(Σ)
    stds = sqrt.(clamp.(vals, 0, Inf))

    ellipse_mat = vecs * nsigma * Diagonal(stds) * _COV_ELLIPSE_CIRCLE .+ μ
    ellipse_points = Point{2,Float64}.(_stats_position.(ellipse_mat[1, :]), _stats_position.(ellipse_mat[2, :]))

    axes_segments = Point{2,Float64}[]
    for i in 1:2
        direction = vecs[:, i] * _stats_extent(stds[i], nsigma)
        p1 = Point{2,Float64}(_stats_position.(μ .- direction))
        p2 = Point{2,Float64}(_stats_position.(μ .+ direction))
        (iszero(stds[i]) || iszero(nsigma) || p1 != p2) || BAT._plot_weight_error("covariance axis is indistinguishable as Float64")
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

# Unsupported covariance arithmetic raises instead of producing an empty result.
function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Cov2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    _stats_validate_nsigma(config.nsigma)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_cov2d_primitives()
    coords, w_prob = si
    (; nsigma) = config
    μ = [_stats_mean(view(coords, 1, :), w_prob), _stats_mean(view(coords, 2, :), w_prob)]
    # Matrix(...): StatsBase.cov(::DenseMatrix, ...) doesn't accept a SubArray.
    Σ = StatsBase.cov(StatsBase.SimpleCovariance(corrected=false), Matrix(coords), w_prob; dims=2, mean=reshape(μ, :, 1))
    all(isfinite, Σ) || BAT._plot_weight_error("weighted covariance is outside the working numerical range")
    for i in axes(coords, 1)
        (Σ[i, i] > 0 || (iszero(Σ[i, i]) && allequal(view(coords, i, :)))) ||
            BAT._plot_weight_error("nonconstant covariance diagonal is outside the working numerical range")
    end
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

    # Swap each point's coordinates, including the covariance-axis endpoints.
    if transposed
        ellipse_points = Point{2,Float64}[Point{2,Float64}(p[2], p[1]) for p in ellipse_points]
        axes_segments = Point{2,Float64}[Point{2,Float64}(p[2], p[1]) for p in axes_segments]
    end

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

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Std1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    _stats_validate_nsigma(config.nsigma)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_std1d_primitives()
    coords, w_prob = si
    (; nsigma) = config
    vals = vec(coords)
    μ = _stats_mean(vals, w_prob)
    σ = _stats_std(vals, w_prob)
    return (positions=_stats_limits(μ, σ, nsigma),)
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

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Std2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    _stats_validate_nsigma(config.nsigma)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_std2d_primitives()
    coords, w_prob = si
    (; nsigma) = config
    x, y = view(coords, 1, :), view(coords, 2, :)
    μ_x, μ_y = _stats_mean(x, w_prob), _stats_mean(y, w_prob)
    σ_x, σ_y = _stats_std(x, w_prob), _stats_std(y, w_prob)
    return (
        x_lines=_stats_limits(μ_x, σ_x, nsigma),
        y_lines=_stats_limits(μ_y, σ_y, nsigma)
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

    transposed && ((x_lines, y_lines) = (y_lines, x_lines))

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
    μ = _stats_mean(vec(coords), w_prob)
    return (x=[_stats_position(μ)],)
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
    μ_x = _stats_mean(view(coords, 1, :), w_prob)
    μ_y = _stats_mean(view(coords, 2, :), w_prob)
    return (μ_x=[_stats_position(μ_x)], μ_y=[_stats_position(μ_y)])
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

    transposed && ((μ_x, μ_y) = (μ_y, μ_x))

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

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Errorbars1D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    _stats_validate_nsigma(config.nsigma)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_errorbars1d_primitives()
    coords, w_prob = si
    (; nsigma) = config
    vals = vec(coords)
    μ = _stats_mean(vals, w_prob)
    σ = _stats_std(vals, w_prob)
    _stats_limits(μ, σ, nsigma)
    return (μ=[_stats_position(μ)], err=[_stats_extent(σ, nsigma)])
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

function compute_plotting_primitives(
    marg_coords::SubArray,
    weights::SubArray,
    recipe::Errorbars2D,
    ::LiveRecipe,
    ::LiveCell,
    config::NamedTuple
)
    _stats_validate_nsigma(config.nsigma)
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_errorbars2d_primitives()
    coords, w_prob = si
    (; nsigma) = config
    x, y = view(coords, 1, :), view(coords, 2, :)
    μ_x, μ_y = _stats_mean(x, w_prob), _stats_mean(y, w_prob)
    σ_x, σ_y = _stats_std(x, w_prob), _stats_std(y, w_prob)
    _stats_limits(μ_x, σ_x, nsigma)
    _stats_limits(μ_y, σ_y, nsigma)
    return (μ_x=[_stats_position(μ_x)], μ_y=[_stats_position(μ_y)], err_x=[_stats_extent(σ_x, nsigma)], err_y=[_stats_extent(σ_y, nsigma)])
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

    # Swap error extents together with their means.
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
    config.npoints_pdf isa Integer && config.npoints_pdf >= 2 ||
        throw(ArgumentError("npoints_pdf must be an integer greater than one"))
    si = _stats_inputs(marg_coords, weights, config)
    isnothing(si) && return _empty_pdf1d_primitives()
    coords, w_prob = si
    (; npoints_pdf) = config

    vals = vec(coords)
    # An exact point mass can have a small positive computed variance because
    # the weighted mean rounds. Detect it before fitting, ignoring zero-weight
    # observations, which do not contribute to the marginal distribution.
    first_positive = findfirst(>(0), w_prob)
    if !isnothing(first_positive) && all(
        i -> iszero(w_prob[i]) || vals[i] == vals[first_positive], eachindex(vals, w_prob)
    )
        return _empty_pdf1d_primitives()
    end
    μ = _stats_mean(vals, w_prob)
    σ = _stats_std(vals, w_prob)
    dist = Normal(μ, σ)
    lo, hi = _stats_limits(μ, σ, 4)
    x = collect(LinRange(lo, hi, npoints_pdf))
    y = map(v -> BAT._plot_float64(v, "Gaussian PDF height"), pdf.(dist, x))

    poly_points = vcat(
        Point{2,Float64}.(x, y),
        Point{2,Float64}.(reverse(x), 0.0)
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
