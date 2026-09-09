# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    struct BATMakieVisualization <: BATVisBackend

*Experimental feature, not yet part of stable public API.*

Configuration of the Makie-based interactive corner-plot visualizer. Pass it
via `BATContext(visualizer = BATVisualizer(BATMakieVisualization()))` to
visualize MCMC sampling live; requires a Makie backend (e.g. GLMakie) to be
loaded. A used visualizer is single-use -- create a fresh one per run.

Constructors:

* ```BATMakieVisualization(; dark = false, max_buffered = 200, adaptive_batching = true, batch_growth_rate = 1.2, trace_nsteps = 20)```
"""
struct BATMakieVisualization <: BATVisBackend
    recipes::NamedTuple
    # Concrete field types: abstract Integer/Real fields would box every
    # access and turn per-flush arithmetic into dynamic dispatch; `new` converts.
    vsel::Vector{Int}
    N_max::Int
    n_batch::Int
    # Backpressure watermark: once this many samples are buffered unflushed,
    # sampling blocks until the listener catches up. With adaptive_batching it
    # (like n_batch) is only the starting value; both grow together at runtime.
    max_buffered::Int
    # When true (default), the effective flush threshold grows geometrically by
    # batch_growth_rate after every flush, fixing an otherwise-quadratic total
    # recompute cost; when false, thresholds stay fixed for maximum live resolution.
    adaptive_batching::Bool
    batch_growth_rate::Float64
    poll_interval::Float64
    dark::Bool
    triagonal_config::NamedTuple
    diagonal_config::NamedTuple

    # N_max fixes the grid size for the visualizer's life; vsel is only the
    # initial selection, applied once at listener startup and never re-read.
    function BATMakieVisualization(recipes, vsel, N_max, n_batch, max_buffered, adaptive_batching, batch_growth_rate, poll_interval, dark, triagonal_config, diagonal_config)
        if length(vsel) > N_max
            @warn "BATMakieVisualization: vsel $vsel has more entries than N_max=$N_max; truncating to $(vsel[1:N_max])."
            vsel = vsel[1:N_max]
        end
        if batch_growth_rate <= 1
            throw(ArgumentError("batch_growth_rate must be > 1 (got $batch_growth_rate); it multiplies the effective batch threshold after every flush"))
        end
        new(recipes, vsel, N_max, n_batch, max_buffered, adaptive_batching, batch_growth_rate, poll_interval, dark, triagonal_config, diagonal_config)
    end
end
export BATMakieVisualization

# Stats-overlay line color; set per-PlotSpec since the overlay plot types'
# automatic color-cycling overrides theme-level colors.
_stats_overlay_color(dark::Bool) = dark ? :white : :black

# The one canonical definition of the default recipe configs -- every consumer
# derives from these two functions via merge, so field sets can't silently drift.
# Credible levels are cdf.(Chi(d), 0:3): the n-sigma mass equivalents in the
# respective dimension, the standard corner-plot convention.
function _default_makie_triagonal_config(; trace_nsteps::Integer = 20, dark::Bool = false)
    # Validated at this single choke point; <= 0 would silently render an empty Trace2D overlay.
    if trace_nsteps < 1
        throw(ArgumentError("trace_nsteps must be >= 1 (got $trace_nsteps); it is how many real MCMC steps back the Trace2D overlay shows"))
    end
    return (
        weights=nothing,
        nsigma=1.0,
        nbins=(100, 100),
        closed=:left,
        normalization=:pdf,
        levels=cdf.(Chi(2), 0:3),
        filter=false,
        colormap=:inferno,
        alpha=1.0,
        rev=false,
        threshold=nothing,
        markersize=2.0,
        # How many real MCMC steps back the Trace2D overlay shows, in actual
        # sampler steps (stepno + weight - 1), so dwells age out at the correct rate.
        trace_nsteps=trace_nsteps,
        # Stats-overlay line color, set per-PlotSpec (not via the theme) because
        # the overlay plot types' automatic color-cycling overrides theme colors.
        stats_color=_stats_overlay_color(dark),
    )
end

function _default_makie_diagonal_config(; dark::Bool = false)
    return (
        weights=nothing,
        nsigma=1.0,
        nbins=100,
        closed=:left,
        normalization=:pdf,
        levels=cdf.(Chi(1), 0:3),
        filter=false,
        colormap=:inferno,
        alpha=1.0,
        y_ebars=0.0,
        filled_pdf=true,
        npoints_pdf=300,
        rev=false,
        # See the triagonal config's stats_color comment above.
        stats_color=_stats_overlay_color(dark),
    )
end

function BATMakieVisualization(; dark::Bool = false, max_buffered::Integer = 4 * 50, adaptive_batching::Bool = true, batch_growth_rate::Real = 1.2, trace_nsteps::Integer = 20)
    recipes = (upper=QuantileHist2D, diagonal=Hist1D, lower=Hist2D)
    vsel = [1, 2, 3] # Default vsel; clamped (with a warning) on its one-time initial application if the posterior has fewer free parameters.
    N_max = 3 # Grid size; cells beyond the (possibly truncated) vsel are simply left dead/empty.
    n_batch = 50 # Flush the buffered samples into the plot once this many have accumulated.
    poll_interval = 0.1 # Seconds between checks of whether a new batch is ready to flush.

    triagonal_config = _default_makie_triagonal_config(trace_nsteps=trace_nsteps, dark=dark)
    diagonal_config = _default_makie_diagonal_config(dark=dark)

    vis_config = BATMakieVisualization(
        recipes,
        vsel,
        N_max,
        n_batch,
        max_buffered,
        adaptive_batching,
        batch_growth_rate,
        poll_interval,
        dark,
        triagonal_config,
        diagonal_config,
    )
    return vis_config
end



"""
    bat_makie_plot(
        samples::DensitySampleVector,
        recipes::NamedTuple = (upper = QuantileHist2D, diagonal = Hist1D, lower = Hist2D),
        vsel::Vector{<:Integer} = [1, 2, 3],
        N_max::Integer = 3;
        dark::Bool = false,
        trace_nsteps::Integer = 20,
        support = nothing
    )

*Experimental feature, not yet part of stable public API.*

Displays an interactive Makie corner-plot figure for `samples`: per-cell
recipe dropdowns, stats/trace overlays, a variable-selection picker and a
step-range slider. Requires a Makie backend to be loaded.

`support` (a measure/prior, or a vector of per-dimension `(lo, hi)` pairs)
enables KDE boundary correction at hard support bounds.
"""
function bat_makie_plot end
export bat_makie_plot


"""
    abstract type BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

Abstract supertype of the Makie corner-plot cell recipes.
"""
abstract type BATMakieRecipe end
export BATMakieRecipe


"""
    struct Hist1D <: BATMakieRecipe

1D marginal histogram (diagonal cells).
"""
struct Hist1D <: BATMakieRecipe end
export Hist1D

"""
    struct Hist2D <: BATMakieRecipe

2D marginal histogram heatmap (off-diagonal cells).
"""
struct Hist2D <: BATMakieRecipe end
export Hist2D

# NAMING, flagged for future reconsideration: all four Quantile* recipes draw
# smallest-interval/HPD credible regions, not quantile-based (central) ones --
# options are a rename (e.g. HPDHist1D) and/or central intervals via split_central.
"""
    struct QuantileHist1D <: BATMakieRecipe

1D histogram with bins colored by smallest-interval (HPD) credible region.
"""
struct QuantileHist1D <: BATMakieRecipe end
export QuantileHist1D

"""
    struct QuantileHist2D <: BATMakieRecipe

2D histogram with cells colored by smallest-interval (HPD) credible region.
"""
struct QuantileHist2D <: BATMakieRecipe end
export QuantileHist2D

"""
    struct Hexbin2D <: BATMakieRecipe

2D hexagonal-binning density (off-diagonal cells).
"""
struct Hexbin2D <: BATMakieRecipe end
export Hexbin2D


"""
    struct Scatter2D <: BATMakieRecipe

2D sample scatter (off-diagonal cells).
"""
struct Scatter2D <: BATMakieRecipe end
export Scatter2D

"""
    struct ChainScatter2D <: BATMakieRecipe

2D sample scatter colored by MCMC chain; only offered when the samples carry
chain identity.
"""
struct ChainScatter2D <: BATMakieRecipe end
export ChainScatter2D


"""
    struct KDE1D <: BATMakieRecipe

1D kernel density estimate (diagonal cells).
"""
struct KDE1D <: BATMakieRecipe end
export KDE1D

"""
    struct KDE2D <: BATMakieRecipe

2D kernel density heatmap (off-diagonal cells).
"""
struct KDE2D <: BATMakieRecipe end
export KDE2D

"""
    struct QuantileKDE1D <: BATMakieRecipe

1D KDE with smallest-interval (HPD) credible bands.
"""
struct QuantileKDE1D <: BATMakieRecipe end
export QuantileKDE1D

"""
    struct QuantileKDE2D <: BATMakieRecipe

2D KDE with smallest-interval (HPD) credible regions.
"""
struct QuantileKDE2D <: BATMakieRecipe end
export QuantileKDE2D


"""
    struct Cov2D <: BATMakieRecipe

Stats overlay: covariance ellipse with principal axes (2D).
"""
struct Cov2D <: BATMakieRecipe end
export Cov2D

"""
    struct Std1D <: BATMakieRecipe

Stats overlay: ±nσ standard-deviation lines (1D).
"""
struct Std1D <: BATMakieRecipe end
export Std1D

"""
    struct Std2D <: BATMakieRecipe

Stats overlay: ±nσ standard-deviation cross lines (2D).
"""
struct Std2D <: BATMakieRecipe end
export Std2D

"""
    struct Mean1D <: BATMakieRecipe

Stats overlay: dashed mean line (1D).
"""
struct Mean1D <: BATMakieRecipe end
export Mean1D

"""
    struct Mean2D <: BATMakieRecipe

Stats overlay: mean crosshair (2D).
"""
struct Mean2D <: BATMakieRecipe end
export Mean2D

"""
    struct Trace2D <: BATMakieRecipe

Recency-colored trace of each chain's last `trace_nsteps` real MCMC steps.
An always-live overlay (`show_trace_upper`/`show_trace_lower`), never in the
recipe dropdown; needs samples with chain and step identity (`MCMCSampleID`).
"""
struct Trace2D <: BATMakieRecipe end
export Trace2D

"""
    struct Errorbars1D <: BATMakieRecipe

Mean with ±nσ error bars (1D).
"""
struct Errorbars1D <: BATMakieRecipe end
export Errorbars1D

"""
    struct Errorbars2D <: BATMakieRecipe

Mean with ±nσ error bars in both variables (2D).
"""
struct Errorbars2D <: BATMakieRecipe end
export Errorbars2D

"""
    struct PDF1D <: BATMakieRecipe

Normal (Gaussian) fit to the 1D marginal.
"""
struct PDF1D <: BATMakieRecipe end
export PDF1D


"""
    bat_theme()::Makie.Theme

*Experimental feature, not yet part of stable public API.*

The BAT light Makie theme used by the corner-plot figure.
"""
function bat_theme end
export bat_theme

"""
    bat_theme_dark()::Makie.Theme

*Experimental feature, not yet part of stable public API.*

The BAT dark Makie theme used by the corner-plot figure.
"""
function bat_theme_dark end
export bat_theme_dark
