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

Calling `empty!(fig)` also disconnects BAT's static control subscriptions. A
retained control no longer updates the old plotting graph after its figure is
emptied.

If static construction or BAT's explicit display fails after allocating a
figure, BAT attempts to release its content and owned display resources before
rethrowing the original error.

For plots with declared step timing, selected repetition counts must be exactly
representable by the samples' weight type. A window that would round a count
raises an error; use integer repetition weights for exact long step ranges.

## Histogram weights and statistical overlays

Histogram `:probability` and `:pdf` normalization and Makie population moments
use temporary relative weights. Common finite weight scaling therefore preserves
their results within the supported numerical range. Stored samples and declared
step counts remain unchanged. Inputs must have finite nonnegative weights and
finite coordinates wherever weight is positive. Positive mass that would be lost
in the working precision raises an error. Empty or zero-mass Makie populations
retain empty-display behavior; a `MarginalDist` requires positive in-bin mass.

Scatter marker areas use relative sample weights, normalized before conversion
to Float64. Finite BigFloat and logarithmic weights therefore retain their
relative sizes when the ratios are representable. Unrepresentable positive
ratios raise an error. `ChainScatter2D` requires chain identity for every sample.

Histogram `:none` retains counts; `:density` divides counts by bin volume without
turning them into probabilities. These modes, report counts and native hexbin
weights require a representable absolute total. Overflow raises an error even
if each bin could individually hold its count. Legacy Plots mean/spread overlays
also retain absolute frequency-weight arithmetic and its correction; a corrected
spread requires total weight greater than one. Only requested overlays and their
selected coordinates are evaluated.

Density division avoids intermediate products of widths and total mass. The
result still must fit the numerical and rendering types actually used. Hist2D
color heights must survive Makie's Float32 conversion; change normalization or
units, or use QuantileHist2D when that limit is reached. ARP weights can reach
this limit even with ordinary coordinate ranges. Statistical overlays
reject nonfinite or collapsed positive spread and interval geometry. Constant
data and singular covariance remain supported where the recipe defines them.

`Hist1D` bar widths come from the original bin edges, including single-bin
histograms. A `BarPlot.width` theme cannot override those widths. Native gap,
dodge, color and label styling remains available. The extension checks the
actual native bar endpoints against exact edge-based layout arithmetic. Each
endpoint's displacement is limited to `32eps(Float32)` times the intended
styled bin width;
intentionally zero-width styling uses the original bin width as its scale.
This bounds relative width error by `64eps(Float32)` for nonzero styled bars,
before backend transforms and rasterization. Gap/dodge styling can intentionally
change the displayed area; it does not change the histogram's stored heights.

Very narrow bins at large coordinate offsets can fail this check even with
finite, distinct edges. Such layouts raise an `ArgumentError`, possibly wrapped
by Makie's compute pipeline, rather than silently shifting or collapsing bins.
Recenter the coordinates or use the original-edge `QuantileHist1D` representation.
Uniformly rescaling coordinates need not improve their offset-to-width ratio.

`Hexbin2D` colors show absolute weighted counts, with thresholds in the same
units. The Makie extension requires Makie 0.24.8 or later in the 0.24 series,
for native Hexbin count nodes and safe transient widget layouts.
After native bin aggregation and threshold filtering, displayed counts
must survive Float32 conversion without materially changing their normalized
color positions. Unsupported magnitudes, collapsed ranges, or insufficient
color resolution raise an `ArgumentError` (possibly wrapped by Makie's compute
pipeline). Empty and uniform-count displays retain Makie's automatic range
policy. Tiny individual weights are allowed when their displayed bin totals
pass this check.

Rescaling count units and thresholds together may resolve magnitude limits;
it need not repair insufficient relative contrast. A normalized histogram or
whole-bin region recipe is another option. Admission bounds scalar color
positions within `8eps(Float32)` under CPU/Float32 arithmetic, including
reciprocal-based normalization and subnormal subtraction flushed to zero. The
reciprocal span must remain normal with headroom for shader arithmetic. This
does not promise distinct pixels for every distinct count or identical palette
interpolation across renderers. Colorbar endpoints retain absolute
count units at Float32 precision.

BAT-owned interactive updates demand these histogram/hexbin checks before
returning to a renderer. Ordinary caller-owned Makie plots may reject on update
or later geometry/color evaluation.
After a rejected update, dispose and recreate the caller-owned figure; in-place
retry is not supported. Live BAT visualization follows its existing failure and
disposal policy and marks the presentation incomplete.

`MarginalDist` checks its dependency's actual PDF and reconstructed bin masses.
Its dependency's cached coordinate moments have separate extreme-range limits.
Histogram regions rank cells by density and accumulate integrated cell mass.
For count/probability histograms the stored heights are masses; for density/PDF
histograms the mass is height times actual bin volume. Regions preserve the
original displayed heights and include the entire cutoff-density plateau. This
intentionally replaces first-bin tie breaking. They are whole-bin highest-density
regions: achieved probability can exceed the request, sometimes substantially,
and the result is not a globally minimum-volume selection of whole bins.

The internal central-region helper uses actual left/right tail masses and retains
a boundary bin when the removed tail would exactly equal its target. Zero requests
are empty, full-mass requests retain every positive cell, and zero-mass histograms
remain empty displays. Makie histogram levels use zero as an inactive sentinel
and support up to three positive requests in `(0,1]`; invalid levels are errors.
Legacy Plots region labels report achieved probability. Custom labels and colors
stay associated with their requests when requests are reordered.

Region arithmetic factors volumes and densities before common power-of-two mass
scaling. Each positive cell must retain positive relative mass; otherwise it raises
an error. Region helpers accept ordinary integer/floating-point histogram storage,
with endpoints exactly representable in the promoted Float64 or BigFloat working
type. Arithmetic must use round-to-nearest and preserve subnormals; BigFloat inputs
at its minimum exponent boundary are rejected. These checks do not change the
caller's arithmetic mode. Finite arithmetic can change a whole boundary plateau
near a probability crossing or nearly tied
computed density; it does not guarantee identical masks under every floating-point
rescaling. Whole-bin regions use original edges, including horizontal fills and
unequal 2D cells.

Legacy Plots contours are approximate isolines of an interpolated density field,
not exact boundaries whose enclosed probability equals the selected cell mass.
Their cutoffs use the same relative-density field that is displayed. Smoothing
uses bin-index units and zero padding/cropping, then recomputes cell probabilities
within that displayed domain; it is not uniform physical-space smoothing on unequal
bins. Contours require at least two cells on each axis and a representable positive
relative-density field, with distinct Float64 centers inside their cells, even when
exact cell regions are representable. Coincident
cutoffs retain the tighter request's color. Numeric contour labels cannot be
interpreted as original density units.

Contour `smoothing` accepts a finite, nonnegative real scalar or a two-tuple
`(sigma_x, sigma_y)`. A scalar applies to both axes; zero disables smoothing on
that axis. Each axis uses Gaussian support radius `2*ceil(sigma)` after conversion
to the Float64 working type. Active smoothing is admitted before constructing its
kernel or padded input, with per-contour limits of `2^20` kernel cells, `2^23`
additional Float64 numeric-buffer slots (64 MiB payload), and `2^26` direct
convolution contributions. Gaussian offset squares must also fit the native
integer type; this further restricts wide one-axis kernels on 32-bit hosts.
The buffer limit includes the kernel, padded input,
coefficient vectors and numerical temporaries; it excludes existing histogram
storage, preceding region preparation, object/runtime overhead and rendering.
It is not a total allocation or process-memory bound. For example, a 200×200
grid admits scalar sigma 9 but rejects sigma 10 on work cost. Excessive requests
raise `ArgumentError` rather than changing the requested smoothing. Reduce sigma
or the number of bins, or select whole-bin regions. Empty/all-zero probability
requests and zero-mass histograms require no smoothing and accept any finite,
nonnegative sigma; sign and finiteness are still validated.

## KDE support and credible regions

For `KDE1D`, `QuantileKDE1D`, `KDE2D` and `QuantileKDE2D`, the `support` keyword
declares hard support in the original sample coordinates. Both
`bat_makie_plot(samples; support=prior, display=false)` and ordinary
`Makie.plot(samples; support=prior)` accept a measure/prior or a vector of
per-dimension `(lo, hi)` pairs. Live plots obtain these bounds from the original
target/prior, including when it is wrapped in an `EvaluatedMeasure`. Unknown
distribution support is conservatively treated as unbounded. Axis limits and
observed extrema do not declare hard support.

Bounds must be strictly ordered, without NaN, incorrectly directed infinities,
or finite endpoints outside Float64's representable range. Positive-weight
observations must be finite and lie inside support; they are never silently
clamped. Weights must be finite and nonnegative. Zero-weight observations are
excluded before bandwidth and extent selection, so a zero-weight outlier cannot
widen an estimate. Empty or all-zero populations have no density to display.
If Float16/Float32 division would discard a positive relative weight that is
representable in Float64, KDE admission raises an error; convert the weights to
Float64 first. Relative weights genuinely below Float64 representation remain
excluded from the convolution population.

Unbounded estimates retain the existing bandwidth rule and fixed FFT grid.
Admission checks the selected bandwidth against a centered, scaled calculation,
the actual Float64 grid and cell edges, convolution arithmetic, and raw density
mass. The Float64 FFT buffer range is checked even when input arithmetic uses
BigFloat. Numerically ambiguous effective sample counts or bandwidths, large offsets
relative to grid spacing, and unsafe coordinate units raise an error before any
visibility mask or credible-region calculation. Recenter coordinates, change
units, or use a histogram. Accepted estimates are not rescaled or replaced by a
different estimator. These checks bound arithmetic fidelity; they do not remove
the inherited fixed-grid approximation or four-bandwidth periodic tails.

Boundary correction uses one Gaussian mirror per nearby finite boundary,
including corner images in 2D, selected within four bandwidths. This is a finite
reflection approximation, not an infinite reflection series. The FFT interval
follows the data and included images, with extra padding; a distant support
endpoint does not stretch the numerical grid. The clipped result is normalized
within its returned interval or cells. Unsupported numerical ranges or a grid
exceeding the internal resolution/memory budget raise an error rather than
returning an unrestricted or silently coarsened density. A histogram is an
alternative for these populations. These limits apply per estimate, not to
total figure memory or rendering latency.

Makie stores numeric heatmap color values as Float32. `KDE2D` therefore rejects
visible densities that would overflow, become zero or subnormal, or leave an
invalid native color span or reciprocal, even if the numerical estimate is valid
in Float64. This includes uniform densities whose automatic range collapses
after Float32 rounding. Change coordinate units or
use `QuantileKDE2D`, which passes credible-region colors directly to Makie.
Density values are never silently rescaled to hide this rendering limit.

In 1D the displayed density is a piecewise-linear curve, with interpolated
endpoints where it meets hard support. `QuantileKDE1D` integrates this curve to
find highest-density regions, including interpolated threshold crossings;
separated modes can produce disconnected filled regions. In 2D the displayed
density consists of constant-density cells with explicit edges inside support.
`QuantileKDE2D` ranks cells by density and accumulates density times actual cell
area. Both region recipes include complete threshold ties, so achieved content
can exceed the requested content. A single constant-density cell belongs to
every nonzero requested region. The ordinary `KDE2D` visibility mask is applied
after normalization and does not redefine probability mass.

Ordinary all-positive unbounded density estimates retain their existing
estimation path where valid. The improved integrated 1D region calculation can
change their credible-region thresholds and polygons. Excluding zero weights
from bandwidth selection is also an intentional correction.

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

Generated coordinate bounds include the original bin intervals of the selected
histogram recipes, in addition to the padded fixed data domain. This preserves
complete outer bins without changing their statistical edges or density heights.
Changing the sample window, including an empty window, leaves these coordinate
bounds unchanged. Switching recipes can change the required bounds and reset
navigation under the rule above. Caller zoom or explicit limits may still crop
bins intentionally.

These rules apply to BAT's interactive figures. Ordinary `Makie.plot` composition
uses the caller's existing Makie layout and reconciliation behavior.

## Live Makie session lifetime

Create a fresh `BATVisualizer(BATMakieVisualization(...))` for each sampling run.
Successful completion drains the final observations and retains an interactive
figure. Closing the live figure disables observation while sampling continues.
To release a retained figure and its observation history, call
`BAT.dispose_visualizer!(vis)` on the thread that created it. This operation
requests disposal without joining active view callbacks. Cleanup waits for
admitted startup, listener and view work, then releases the session's graph and
computed caches without requiring status polling. It also releases a fresh
session that has never created a figure; disposed sessions remain single-use.
Inspect `BAT.visualizer_status(vis).cleanup_pending` if needed. A cleanup fault
keeps release pending; another owner-thread disposal call can retry it.
BAT's GLMakie display, renderer warmup and live initialization must run on
Julia's main thread. BAT creates a separate native screen for each newly
displayed figure, preserving unrelated windows, and reuses that figure's open
screen on repeated display. BAT scopes its theme to figure construction, before
renderer startup. Custom renderer callbacks must not wait on locks held by
their caller, including an outer caller-owned `with_theme` scope.

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

Live plots use the original model's flattened value shape, including when the
sampler's transformation changes the number of coordinates. For example, a
three-component Dirichlet displays all three components although sampling uses
two unconstrained coordinates. Declared constant fields do not occupy plot
dimensions. This also applies when no initialization samples are retained;
initializing an empty display does not add observations or sample weight.

Live figures identify this population with the caption “Observed sampling
history” and the note “May include initialization and burn-in”. The caption remains
visible when controls are collapsed and in exports of the complete Figure.
Applying a range changes the displayed window within that history; it does not
select a sampling phase. Static plots of supplied samples make no such provenance
claim.

After successful startup, a failure in observation or rendering disables further
observation without changing fixed-work samples, computational RNG states, or
public sampling callbacks. Configuration, reuse, and startup errors still throw;
sampling errors and intentional interruption remain errors. Time-limited runs
remain sensitive to observation overhead. Direct scalar iteration does not own
the session and does not promise a final display flush on cancellation/time-limit
exits.
BAT-created observation tasks use private RNG state, preserving the initiating
task's ordinary draws and the streams of its future child tasks. BAT's GL shader
warmup renders without a background loop. Its explicit GL display starts the
renderer through a private startup task, joined before returning or cleaning up
an interrupted startup. Synchronous scene and inline-display callbacks still
execute on the caller, and retain their random draws and child tasks. Renderer
callbacks execute in the renderer task. BAT does not rewind a whole callback or
display call to hide its effects; private renderer seeds do not promise unique
or statistically independent streams for custom rendering code.
Direct caller-owned Makie/GLMakie display calls, including later displaying a
figure returned with `display=false`, are outside this BAT startup guarantee.
Automatic garbage-collection finalizers are also outside the tested boundary.
Initial observation transforms run sequentially on the sampling owner to avoid
launching additional row-processing tasks. Custom transforms, including shape
and Jacobian probes, retain their own random draws and task creation; their
effects are outside the observation neutrality guarantee.
Runtime-owned lazy scheduler initialization is also outside this guarantee:
on Julia 1.13.2, a thread's first blocking wait can itself change future child
streams, even without BAT observation.
Failed or aborted live views are disposed; their outcome remains available
through the status query and diagnostics independently of the renderer.
Disposal after the sampling owner has acquired the session also permits that
same acquisition to finish numerical initialization and continue sampling without
starting a view. Observation history may already have been released at that point.
This continuation is single-use and restricted to the acquiring task and thread;
a disposed session cannot be acquired for a new run. Numerical failures and
non-benign startup failures still propagate.

Cancellation of an MCMC chain group is cooperative. If its sampling owner is
interrupted or a chain fails, BAT requests cancellation and releases participating
observation waits, then joins every launched chain worker before propagating the
error. Initialization timing cleanup consequently follows the last chain writer.
Repeated catchable interrupts during this cleanup do not abandon workers or
replace the original owner exception. An already-entered numerical step or user
hook must finish or throw before its worker can stop; there is no safe forced
termination of arbitrary native model code. This ownership guarantee concerns
BAT's MCMC chain groups, not arbitrary tasks started by user code or separate
parallel sample transformations. It relies on a functioning Julia runtime and
the application's signal policy; it cannot cover process termination.

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
failure; `cleanup_error` records a separate cleanup fault. Each diagnostic is an
inert `(type, message, phase)` snapshot, with a bounded copied type name and
message. Plain `ErrorException` and `ArgumentError` string messages are retained
within the bound; other exceptions use a fallback message. Status and cleanup
never call custom error formatters. Diagnostic snapshots do not retain raw
exceptions, backtraces or failed tasks. Pending cleanup still owns the tasks it
must join. Existing error logging can expose the original exception transiently.
Failed sessions do not
retry detached batches or resume as if observation had remained complete.

`history_released` becomes true only after disposal has released all
session-owned observation history and the cleanup task has terminated
successfully. It remains false after successful sampling until disposal, and
while a requested release is pending. A no-op or provenance-only visualizer has
no such history and reports true. `quiesced` describes observer resources, not
completion of numerical sampling. Retaining only the borrowed, emptied figure
does not retain BAT's observation history. Independently retained graph nodes,
controls, sampling results, user configuration or logger records can still own
data; disposal does not invalidate those external references.
