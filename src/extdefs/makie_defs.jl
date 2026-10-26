# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct BATMakieVisualization <: BATVisBackend

*Experimental feature, not yet part of stable public API.*

Configure a live Makie corner plot for MCMC sampling. Pass the configuration to
`BATVisualizer` and then to `BATContext(visualizer = vis)`. Creating the visualizer
requires Makie; displaying it requires a rendering backend such as GLMakie.
Create a fresh visualizer for each sampling run.

Constructors:

```julia
BATMakieVisualization(;
    dark = false, max_buffered = 200, adaptive_batching = true,
    batch_growth_rate = 1.2, trace_nsteps = 20, observer_timeout = 60.0
)

BATMakieVisualization(
    recipes, vsel, N_max, n_batch, max_buffered, adaptive_batching,
    batch_growth_rate, poll_interval, dark, triagonal_config, diagonal_config;
    observer_timeout = 60.0
)

BATMakieVisualization(
    recipes, vsel, N_max, n_batch, max_buffered, adaptive_batching,
    batch_growth_rate, poll_interval, dark, triagonal_config, diagonal_config,
    observer_timeout
)
```

The keyword constructor accepts only the six keywords shown. It fixes
`vsel = [1, 2, 3]`, `N_max = 3`, `n_batch = 50`, `poll_interval = 0.1` and
the default recipes/configurations below. Use a positional constructor to supply
those settings. `trace_nsteps` is an integer of at least one; it sets
`triagonal_config.trace_nsteps` and is not a separate field.

Fields:

- `recipes::NamedTuple`: initial `(upper, diagonal, lower)` recipe types.
  The keyword constructor uses `(upper = QuantileHist2D, diagonal = Hist1D,
  lower = Hist2D)`.
- `vsel::Vector{Int}`: initial variable indices. Construction truncates a selection
  longer than `N_max`; startup removes indices outside the available coordinates.
  The initial selection is applied once. Default: `[1, 2, 3]`.
- `N_max::Int`: fixed positive grid size. Default: `3`.
- `n_batch::Int`: initial flush threshold in completed walker-steps across
  all chains. Default: `50`.
- `max_buffered::Int`: initial backpressure threshold in the same units.
  Require `0 < n_batch <= max_buffered`; this is not a strict memory limit.
  Default: `200`.
- `adaptive_batching::Bool`: whether the effective flush and backpressure
  thresholds grow after each flush. Default: `true`.
- `batch_growth_rate::Float64`: finite growth factor greater than one,
  required even when adaptive batching is disabled. Default: `1.2`.
- `poll_interval::Float64`: finite positive polling interval in seconds.
  Default: `0.1`.
- `dark::Bool`: select the BAT dark theme for the live figure. Default: `false`.
- `triagonal_config::NamedTuple`: off-diagonal recipe and overlay settings.
  The keyword constructor supplies BAT defaults, including `trace_nsteps = 20`
  unless overridden, and a statistics color chosen for `dark`.
- `diagonal_config::NamedTuple`: diagonal recipe and overlay settings.
  The keyword constructor supplies BAT defaults with a statistics color chosen
  for `dark`.
- `observer_timeout::Float64`: finite positive timeout in seconds for each
  continuous backpressure wait and, separately, finalization. Default: `60.0`.

Defaults listed above are those of the keyword constructor. Positional values
are converted to the stored field types and validated. Timeout expiry disables
observation while sampling continues; it requires scheduler progress and cannot
cancel a stuck native renderer. Inspect [`BAT.visualizer_status`](@ref), and use
[`BAT.dispose_visualizer!`](@ref) to release a retained completed view.

See [Live Makie configuration](@ref) for batching, timing and RNG guarantees,
and [Live Makie session lifetime](@ref) for figure ownership and cleanup.
"""
struct BATMakieVisualization <: BATVisBackend
    recipes::NamedTuple
    vsel::Vector{Int}
    N_max::Int
    n_batch::Int
    max_buffered::Int
    adaptive_batching::Bool
    batch_growth_rate::Float64
    poll_interval::Float64
    dark::Bool
    triagonal_config::NamedTuple
    diagonal_config::NamedTuple
    observer_timeout::Float64

    function BATMakieVisualization(
        recipes, vsel, N_max, n_batch, max_buffered, adaptive_batching, batch_growth_rate,
        poll_interval, dark, triagonal_config, diagonal_config, observer_timeout
    )
        N_max = Int(N_max)
        N_max > 0 || throw(ArgumentError("N_max must be positive (got $N_max)"))
        n_batch = Int(n_batch)
        max_buffered = Int(max_buffered)
        batch_growth_rate = Float64(batch_growth_rate)
        poll_interval = Float64(poll_interval)
        observer_timeout = Float64(observer_timeout)
        isfinite(poll_interval) && poll_interval > 0 ||
            throw(ArgumentError("poll_interval must be finite and positive (got $poll_interval)"))
        isfinite(observer_timeout) && observer_timeout > 0 ||
            throw(ArgumentError("observer_timeout must be finite and positive (got $observer_timeout)"))
        if !(0 < n_batch <= max_buffered)
            throw(ArgumentError("Require 0 < n_batch <= max_buffered (got n_batch=$n_batch, max_buffered=$max_buffered) so producers cannot block before a flush is possible"))
        end
        if length(vsel) > N_max
            @warn "BATMakieVisualization: vsel $vsel has more entries than N_max=$N_max; truncating to $(vsel[1:N_max])."
            vsel = vsel[1:N_max]
        end
        if !(isfinite(batch_growth_rate) && batch_growth_rate > 1)
            throw(ArgumentError("batch_growth_rate must be finite and > 1 (got $batch_growth_rate); it multiplies the effective batch threshold after every flush"))
        end
        new(
            recipes, vsel, N_max, n_batch, max_buffered, adaptive_batching,
            batch_growth_rate, poll_interval, dark, triagonal_config, diagonal_config,
            observer_timeout
        )
    end
end
export BATMakieVisualization

# Preserve the original positional constructor, with a named timeout option.
function BATMakieVisualization(
    recipes, vsel, N_max, n_batch, max_buffered, adaptive_batching, batch_growth_rate,
    poll_interval, dark, triagonal_config, diagonal_config; observer_timeout::Real = 60.0
)
    BATMakieVisualization(recipes, vsel, N_max, n_batch, max_buffered, adaptive_batching,
        batch_growth_rate, poll_interval, dark, triagonal_config, diagonal_config, observer_timeout)
end


# BATVisualizer(::BATMakieVisualization) needs the Makie extension to build its
# live content; see BAT._vis_required_pkg.
_vis_required_pkg(::BATMakieVisualization) = :Makie

# Stats-overlay line color; set per-PlotSpec since the overlay plot types'
# automatic color-cycling overrides theme-level colors.
_stats_overlay_color(dark::Bool) = dark ? :white : :black

# Shared recipe defaults. Credible levels use the n-sigma mass equivalents for each dimension.
function _default_makie_triagonal_config(; trace_nsteps::Integer = 20, dark::Bool = false)
    if trace_nsteps < 1
        throw(ArgumentError("trace_nsteps must be >= 1 (got $trace_nsteps); it is how many real MCMC steps back the Trace2D overlay shows"))
    end
    return (
        weights = nothing,
        nsigma = 1.0,
        nbins = (100, 100),
        closed = :left,
        normalization = :pdf,
        levels = cdf.(Chi(2), 0:3),
        filter = false,
        colormap = :inferno,
        alpha = 1.0,
        rev = false,
        threshold = nothing,
        markersize = 2.0,
        # Trace2D's window in explicit cycle-aware walker-step spans;
        # repetition weights alone do not establish elapsed-step timing.
        trace_nsteps = trace_nsteps,
        stats_color = _stats_overlay_color(dark),
    )
end


function _default_makie_diagonal_config(; dark::Bool = false)
    return (
        weights = nothing,
        nsigma = 1.0,
        nbins = 100,
        closed = :left,
        normalization = :pdf,
        levels = cdf.(Chi(1), 0:3),
        filter = false,
        colormap = :inferno,
        alpha = 1.0,
        y_ebars = 0.0,
        filled_pdf = true,
        npoints_pdf = 300,
        rev = false,
        stats_color = _stats_overlay_color(dark),
    )
end


function BATMakieVisualization(;
    dark::Bool = false, max_buffered::Integer = 4 * 50, adaptive_batching::Bool = true,
    batch_growth_rate::Real = 1.2, trace_nsteps::Integer = 20, observer_timeout::Real = 60.0
)
    recipes = (upper = QuantileHist2D, diagonal = Hist1D, lower = Hist2D)
    vsel = [1, 2, 3]
    N_max = 3
    n_batch = 50
    poll_interval = 0.1

    triagonal_config = _default_makie_triagonal_config(trace_nsteps = trace_nsteps, dark = dark)
    diagonal_config = _default_makie_diagonal_config(dark = dark)

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
        observer_timeout,
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
        display::Bool = true,
        trace_nsteps::Integer = 20,
        support = nothing,
        step_intervals = nothing
    )

*Experimental feature, not yet part of stable public API.*

Return a `Makie.Figure` containing an interactive corner plot for `samples`,
with recipe menus, statistics/trace overlays, variable selection and Apply/All
range controls. Requires Makie. The BAT light or dark theme is scoped to figure
construction; unrelated figures and the global theme are unchanged.

By default, explicitly display the same figure before returning it. Set
`display=false` to defer that display and optional shader warmup; the figure and
controls are still constructed. Displaying or saving requires a loaded rendering
backend. This keyword does not suppress automatic frontend display of the return
value; use a trailing semicolon for that. For examples and failure behavior, see
[Interactive static Makie figures](@ref).

Without `step_intervals`, the range selects stored rows per walker and preserves
their weights. Explicit repetition timing enables elapsed-step ranges and traces;
bare sample metadata and statistical weights, including ARP weights, do not
certify dwell durations. See [Sample ranges and repetition timing](@ref) for the
per-row interval format, validity requirements and exact range controls.

`support` declares hard KDE bounds in original sample coordinates, as a
measure/prior or a vector of per-dimension `(lo, hi)` pairs. Invalid bounds or
positive-weight observations outside support raise an error rather than being
clamped. See [KDE support and credible regions](@ref) for normalization,
zero-weight handling, approximation and numerical limits; use a histogram when
a population cannot be represented by the supported KDE calculation.

Calling `empty!(fig)` releases the static figure's content and BAT control
subscriptions. Failed construction or explicit display attempts owned-resource
cleanup before rethrowing the original error. Ordinary caller-owned composition
is documented under [Ordinary Makie composition](@ref).
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

*Experimental feature, not yet part of stable public API.*

1D marginal histogram (diagonal cells).
"""
struct Hist1D <: BATMakieRecipe end
export Hist1D

"""
    struct Hist2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

2D marginal histogram heatmap (off-diagonal cells).
"""
struct Hist2D <: BATMakieRecipe end
export Hist2D

# NAMING, flagged for future reconsideration: all four Quantile* recipes draw
# smallest-interval/HPD credible regions, not quantile-based (central) ones --
# options are a rename (e.g. HPDHist1D) and/or central intervals via split_central.
"""
    struct QuantileHist1D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

1D histogram with whole bins colored by highest-density credible region.
Cells are ranked by density; coverage integrates their actual widths. Complete
threshold plateaus are included, so achieved probability can exceed the request.
This is a whole-bin approximation, not a minimum-width subset optimization.
Zero is an inactive level sentinel; up to three positive levels in `(0, 1]`
are supported, including full-mass regions at one.
"""
struct QuantileHist1D <: BATMakieRecipe end
export QuantileHist1D

"""
    struct QuantileHist2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

2D histogram with whole cells colored by highest-density credible region.
Cells are ranked by density and accumulated by actual cell probability. Complete
threshold plateaus are included. Original cell edges determine the colored area.
See [`QuantileHist1D`](@ref) for level and whole-bin approximation conventions.
"""
struct QuantileHist2D <: BATMakieRecipe end
export QuantileHist2D

"""
    struct Hexbin2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

2D hexagonal binning colored by absolute weighted counts (off-diagonal cells).
Thresholds use the same count units. Displayed bins must retain their normalized
color positions through Makie's Float32 conversion; unsupported magnitudes or
insufficient color resolution raise an error. Dispose and recreate a caller-owned
figure after a rejected update.
"""
struct Hexbin2D <: BATMakieRecipe end
export Hexbin2D


"""
    struct Scatter2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

2D sample scatter (off-diagonal cells).
"""
struct Scatter2D <: BATMakieRecipe end
export Scatter2D

"""
    struct ChainScatter2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

2D sample scatter colored by MCMC chain; only offered when the samples carry
chain identity.
"""
struct ChainScatter2D <: BATMakieRecipe end
export ChainScatter2D


"""
    struct KDE1D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

1D kernel density estimate (diagonal cells), represented by a piecewise-linear
curve. With declared hard support, the returned curve stays inside support and
has unit trapezoid integral. Boundary correction uses finite Gaussian reflections.
"""
struct KDE1D <: BATMakieRecipe end
export KDE1D

"""
    struct KDE2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

2D kernel density heatmap (off-diagonal cells), represented by constant-density
cells with explicit edges. With declared hard support, complete cells stay inside
support and their density times area sums to one before the visibility mask.
Makie stores numeric heatmap colors as Float32; a visible density that would
overflow or become zero raises an error. Change coordinate units or use
`QuantileKDE2D`, whose colors represent credible regions directly.
"""
struct KDE2D <: BATMakieRecipe end
export KDE2D

"""
    struct QuantileKDE1D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

1D KDE with highest-density credible regions. Region contents integrate the
piecewise-linear curve, including interpolated threshold crossings. Separated
modes can produce disconnected regions; complete tied plateaus are included.
"""
struct QuantileKDE1D <: BATMakieRecipe end
export QuantileKDE1D

"""
    struct QuantileKDE2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

2D KDE with highest-density credible regions. Cells are ranked by density and
their probabilities use actual cell areas. Complete threshold ties are included,
so achieved content may exceed the requested content, including for a single cell.
"""
struct QuantileKDE2D <: BATMakieRecipe end
export QuantileKDE2D


"""
    struct Cov2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

Stats overlay: covariance ellipse with principal axes (2D).
"""
struct Cov2D <: BATMakieRecipe end
export Cov2D

"""
    struct Std1D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

Stats overlay: ±nσ standard-deviation lines (1D).
"""
struct Std1D <: BATMakieRecipe end
export Std1D

"""
    struct Std2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

Stats overlay: ±nσ standard-deviation cross lines (2D).
"""
struct Std2D <: BATMakieRecipe end
export Std2D

"""
    struct Mean1D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

Stats overlay: dashed mean line (1D).
"""
struct Mean1D <: BATMakieRecipe end
export Mean1D

"""
    struct Mean2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

Stats overlay: mean crosshair (2D).
"""
struct Mean2D <: BATMakieRecipe end
export Mean2D

"""
    struct Trace2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

Recency-colored trace of each walker's last `trace_nsteps` steps in the latest
cycle contributing to the selected window. Paths are split at gaps and never
connect different cycles or walkers. An always-live overlay
(`show_trace_upper`/`show_trace_lower`), never in the recipe dropdown; requires
captured repetition timing or explicit `step_intervals` for static plots.
"""
struct Trace2D <: BATMakieRecipe end
export Trace2D

"""
    struct Errorbars1D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

Mean with ±nσ error bars (1D).
"""
struct Errorbars1D <: BATMakieRecipe end
export Errorbars1D

"""
    struct Errorbars2D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

Mean with ±nσ error bars in both variables (2D).
"""
struct Errorbars2D <: BATMakieRecipe end
export Errorbars2D

"""
    struct PDF1D <: BATMakieRecipe

*Experimental feature, not yet part of stable public API.*

Normal (Gaussian) fit to the 1D marginal.
"""
struct PDF1D <: BATMakieRecipe end
export PDF1D


"""
    bat_theme()::Makie.Theme

*Experimental feature, not yet part of stable public API.*

The BAT light Makie theme used by the corner-plot figure.

`bat_makie_plot` and the live visualizer apply it to their own figures only,
via a scoped `with_theme`. To use it for every Makie plot in a session, apply
it yourself with `Makie.set_theme!(bat_theme())`.
"""
function bat_theme end
export bat_theme

"""
    bat_theme_dark()::Makie.Theme

*Experimental feature, not yet part of stable public API.*

The BAT dark Makie theme used by the corner-plot figure. Scoped to BAT's own
figures like [`bat_theme`](@ref); apply it with `Makie.set_theme!` for a
session-wide dark theme.
"""
function bat_theme_dark end
export bat_theme_dark
