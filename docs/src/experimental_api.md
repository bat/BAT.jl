# Experimental features

These are experimental features. Forward/backward compatibility does *not*
follow [Julia's semantic versioning rules](https://julialang.github.io/Pkg.jl/v1/compatibility/).
Instead, compatibility is only guaranteed across changes in patch version, but
*not* across changes of minor (or major) version.

The features listed here are likely to transition to the stable API in future
versions, but may still evolve in a API-breaking fashion during that process.

```@docs
ARPWeighting
BAT.batalgorithm
bat_compare
bat_integrated_autocorr_len
bat_marginalmode
BAT.auto_renormalize
BAT.BinnedModeEstimator
BAT.convert_for
BAT.DistributionTransform
BAT.LowRankAffineTransform
BAT.PathfinderTransformInit
BAT.enable_error_log
BAT.error_log
BAT.EvalException
BAT.evalmeasure_impl
BAT.ext_default
BAT.get_adselector
BAT.get_valid_adselector
BAT.PackageExtension
BAT.pkgext
BAT.set_rng
batmeasure
BridgeSampling
EllipsoidalNestedSampling
GridSampler
HierarchicalDistribution
PriorImportanceSampler
ReactiveNestedSampling
SobolSampler
truncate_batmeasure
ValueAndThreshold

BAT.validate_evalmeasure
BAT.MCMCChainState
BAT.MCMCChainStateInfo
BAT.MCMCIterator
BAT.MCMCProposal
BAT.MCMCProposalState
BAT.MCMCProposalTunerState
BAT.MCMCState
BAT.MCMCTempering
BAT.MCMCTransformTunerState
BAT.MeasureEvalInfo
BAT.PolarShellDistribution
BAT.SimpleMCMCProposalState
BAT.TemperingState

bat_makie_plot
bat_theme
bat_theme_dark
BAT.BATVisBackend
BATMakieRecipe
BATMakieVisualization
BATVisualizer
BAT.visualizer_figure
BAT.visualizer_status
BAT.dispose_visualizer!
ChainScatter2D
Cov2D
Errorbars1D
Errorbars2D
Hexbin2D
Hist1D
Hist2D
KDE1D
KDE2D
Mean1D
Mean2D
PDF1D
QuantileHist1D
QuantileHist2D
QuantileKDE1D
QuantileKDE2D
Scatter2D
Std1D
Std2D
Trace2D
```

## Interactive static Makie figures

`bat_makie_plot` returns the `Makie.Figure` it constructs, including its controls.
By default it also displays that same figure:

```julia
fig = bat_makie_plot(samples);
```

To defer display, for example when saving from a script, use:

```julia
fig = bat_makie_plot(samples; display=false);
save("samples.png", fig)
```

`display=false` skips BAT's explicit display and optional shader warmup. It still
constructs the Makie figure and evaluates the plotting graph; saving or displaying
later requires a suitable backend. It does not suppress automatic display of the
returned value by a REPL, notebook, or IDE. The semicolons above suppress that
extra output. For automatic inline output, use a final expression such as
`bat_makie_plot(samples; display=false)` without a semicolon.

The return changed from `nothing` to `Makie.Figure`. Existing explicit display is
preserved, but return-value checks, automatic frontend output, and collections
of results can change behavior or retain figures. This is an experimental API
change, subject to the compatibility policy at the top of this page.

The result supports normal Makie figure operations. Its internal BAT graph and
control callbacks are not public API. Interactive grid updates follow the state
policy below. Ordinary `Makie.plot` composition retains its
existing interface; the `display` keyword above belongs to `bat_makie_plot`.
Live sessions retain the separate lifetime operations described below.

## Interactive grid updates

Interactive static and live figures keep their axes in fixed grid cells while
recipes, samples and selected variables change. Inactive cells are collapsed;
recipe round trips retain Figure membership and working controls. Plot object
identity and the nesting of recipe internals are implementation details.

BAT owns the grid's labels, decorations, aspect and computed bounds. Other axis
styling, such as titles and background colors, stays with its grid position.
Across BAT updates, navigation and explicit `limits!` overrides survive while
that cell represents the same variables with unchanged finite computed bounds.
A variable/activation change or changed bounds resets the whole cell, including
both coordinates when a diagonal density height changes. Automatic dimensions
are recomputed after updates. Removed variables have no saved navigation state.
Makie's display and export operations can independently reset pan/zoom; this
policy does not override those operations.

These rules apply to BAT's interactive figures. Ordinary `Makie.plot` composition
uses the caller's existing Makie layout and reconciliation behavior.

## Live Makie session lifetime

Create a fresh `BATVisualizer(BATMakieVisualization(...))` for each sampling run.
Successful completion drains the final observations and retains an interactive
figure. Closing the live figure disables observation while sampling continues.
To release a retained figure, call `BAT.dispose_visualizer!(vis)` on the thread
that created it. This operation requests disposal without joining active view
callbacks; inspect `BAT.visualizer_status(vis).cleanup_pending` if needed.
Live GLMakie initialization must run on Julia's main thread.

Keep the original visualizer to retrieve its completed figure:

```julia
vis = BATVisualizer(BATMakieVisualization())
context = BATContext(visualizer=vis)
result = bat_sample(target, algorithm, context)
fig = BAT.visualizer_figure(vis);
if !isnothing(fig)
    save("live-samples.png", fig)
end
```

The getter returns the same retained `Makie.Figure` once observation has completed
successfully and its tasks have settled. It returns `nothing` during sampling,
inside view callbacks, while the view is busy, or when no usable completed figure
remains. It does not wait, create a view, display or render. Obtain and use the
figure on the thread that created the session; a wrong-thread query raises
`ArgumentError` when the session can be inspected. Control-lock contention
returns `nothing` without waiting.

This is a borrowed reference. Coordinate saving or customization with view
callbacks and disposal; a later close or `BAT.dispose_visualizer!(vis)` can empty
the same figure. Retrieval does not transfer ownership or protect subsequent
operations. A deep-copied visualizer, including the sampling result's provenance
context, has no live resources and returns `nothing`. The figure retains the
existing live observation population, which can include initialization and
burn-in; it is not a reconstructed plot of the returned posterior samples.
Availability follows the current completion and view state; an older diagnostic
alone does not prevent retrieval. Use `BAT.visualizer_status(vis)` to inspect it.

After successful startup, a failure in observation or rendering disables further
observation without changing fixed-work samples, computational RNG states, or
public sampling callbacks. Configuration, reuse, and startup errors still throw;
sampling errors and intentional interruption remain errors. Time-limited runs
remain sensitive to observation overhead. Direct scalar iteration does not own
the session and does not promise a final display flush on cancellation/time-limit
exits.
Failed or aborted live views are disposed; their outcome remains available
through the status query and diagnostics independently of the renderer.

`observer_timeout=60.0` bounds each continuous capacity wait and, separately, the
owner's shutdown wait. Nested or repeated cleanup shares the same allowance
until the requested resources settle. Expiry stops observation and reports
incompleteness; it does not cancel sampling. The allowance uses monotonic time and requires Julia's
scheduler to make progress. It cannot interrupt a non-yielding native renderer.
A listener still running after a timeout stays owned and reports
`quiesced=false` and `cleanup_pending=true` until it actually finishes.

`BAT.visualizer_status(vis)` is independent of rendering. `outcome` describes the
run, `data_complete` describes confirmed observation history, and
`last_publication` counts successfully returned data publications. Closing a
completed figure preserves its historical completeness. A later view error may
invalidate `presentation_valid` independently. `error` preserves the first
failure; `cleanup_error` records a separate cleanup fault. Failed sessions do not
retry detached batches or resume as if observation had remained complete.
