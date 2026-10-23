# Optional Makie checks

From the repository root:

```sh
julia --project=test/optional/makie -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --threads=2 --project=test/optional/makie test/optional/makie/runtests.jl
```

Explicit `Pkg.develop` selects the checkout being tested, including on Julia
1.10. Each case runs in a fresh process. Its parent enforces a 600-second deadline
and preserves stdout/stderr in the printed log directory. An expired case fails
the run; killing it does not count as successful lifecycle cleanup.
`BAT_MAKIE_CASE_TIMEOUT` can change the process allowance for slow cold builds.
Pass case names to run a subset, for example `lifecycle_timeout buffering`.

These CairoMakie cases exercise live BAT entry points, real sample transport,
fault boundaries, rendering, and existing numerical regressions. CI runs them on
Julia 1.10 and the current stable Julia. The ordinary core suite separately tests
backend-neutral ownership and pre-join stop signalling without loading Makie.
The `lifecycle_owner` case interrupts a real MCMC group while a producer is
blocked in Makie transport, checks that both numerical workers finish before the
owner propagates the original interruption, and keeps the listener blocked until
after those assertions. The `core` case includes the backend-neutral owner and
RNG/sample-preservation tests. Real process-SIGINT tests are separately bounded
by [the optional MCMC owner runner](../mcmc_owner/README.md).

`gl_smoke.jl` additionally tests native GLMakie recipe/selection round trips,
rendering, and live close isolation from an unrelated screen.
Run it with an external process deadline in a scratch environment containing
this checkout and GLMakie, on a machine with a working display/OpenGL context.
Each case creates two hidden native windows and closes only those it owns.
This manual check is separate from the headless CairoMakie CI job.

The `static_result` case checks the public interactive figure return, explicit
display and deferred rendering, weighted histogram contents, working controls,
independent figures, and scoped themes. `precompile_workload` and
`precompile_failure` exercise the real workload and its owned-figure cleanup on
success and injected failure. These cases also run in separate bounded processes.
The precompile cases add narrow test-only methods at real workload boundaries;
keep them isolated rather than including them in a shared test process.
They verify real Apply/All graph updates and reject swallowed live-widget
failures. `warmup` checks renderer bookkeeping, retries and temporary-Figure
cleanup with Cairo rendering and a second synthetic renderer identity.
`gl_warmup.jl` supplies the native Cairo → GL → Cairo check, including survival
of an unrelated GL window. Like `gl_smoke.jl`, run it separately with an external
deadline and a working display; it is not part of headless CI.

The `domain` case checks live bounds against analytic and pre-change oracles,
including native finite filtering and warmed allocation growth with fixed walker
topology. `domain_publication` checks those bounds through real ingestion and
flushes, including replacement of pending extremes and prior-generation ownership.

`display_schema`, `display_live`, and `display_failures` cover original-coordinate
storage when transforms change dimensions, nested shapes and constants, empty
startup and idle slots, Float32 storage, and transformed domain bounds. They
check explicit timing/mass oracles, pending replacement, public fixed-work
sample/callback/RNG parity, real variable selections, and owned cleanup after
startup or later conversion failures. Core sampling and transformation behavior
is compared with no-observer controls; these tests do not change it.
`gl_display_schema.jl` renders the third Dirichlet component in native GL windows
with both retained and empty startup, checks live selection round trips and
sample/RNG parity, and disposes each session. Run it separately with the same
native-display and external-deadline requirements as `gl_smoke.jl`.

The `lifecycle_figure` case checks completed live figure retrieval through public
sampling, saving and controls, plus owning-thread checks, nonwaiting guards,
callback reentrancy, actual disposal, and focused lifecycle boundary fixtures.

The `history_release`, `history_diagnostics`, and `release_lifecycle` cases check
whole-history disposal, including collection without status polling, retained
borrowed figures and external graph/control aliases, bounded inert diagnostics,
startup/listener/view barriers, failed cleanup retries and task ownership.
Weak-reference checks cover mutable graph/cache arrays, with explicit external
aliases as negative controls; they do not impose process RSS thresholds.

The `view_state` and `lifecycle_grid` cases check interactive grid membership,
recipe/variable transitions, plot contents, controls, navigation reset rules and
subscription cleanup, including empty/automatic-limit cells. The adapter uses
Makie's scene-level plot-spec updater; the ordinary caller-owned SpecApi path
has separate behavior.

The `api_boundaries` case covers initial picker clamping, positive integer grid
sizes, heterogeneous sample metadata, exact timed-weight conversion, and static
control subscription cleanup. `lifecycle_interactive_interrupt` injects an
interrupt through a real control callback and checks observation status,
disposal, and continued sampler/RNG equality.
The `numerics` case checks Float64 geometry, zero-mass KDE behavior, stable
relative scatter weights, and actual PDF/covariance rendering at extreme scales.
The `weights` case checks operation-specific histogram and population-moment
arithmetic against independent wider-precision oracles, absolute count errors,
renderer conversion limits, real graphs and both public static entry paths.
`weight_failure` triggers a real numerical overlay error during public sampling,
checks exact fixed-work samples, callback snapshots, chain state and RNG parity,
and verifies observation failure and autonomous history release.

`scatter_admission` checks relative marker areas with finite BigFloat, BigInt
and logarithmic weights before renderer conversion, plus rejection of missing
chain metadata. `failure_ownership` checks partial static construction and
display rollback, unrelated-figure survival, and grid/bin dispatch with several
integer widths. The dispatch checks do not replace a native 32-bit platform run.

`bounded_kde` compares supported KDEs with independent reflected Gaussian
mixtures, checks concentrated and narrow-support densities, actual probability
geometry, coordinate/weight scaling and explicit numerical/resource limits.
`kde_integration` exercises all four KDE recipes through static, ordinary and
live entry points, original-target support forwarding, explicit rendered edges,
transposition and empty/window/support transitions. The precompile workload uses
truthful bounded samples so strict support checks run during extension building.

`histogram_regions` checks integrated cell probabilities, plateau colors,
original-edge polygons and heatmaps, transpose/empty transitions, invalid levels,
and actual Cairo output. These checks use independent unequal-cell fixtures.

`histogram_viewport` checks that fixed-domain histogram edges fit generated
limits, including coarse/asymmetric bins, constant and near-ULP coordinates,
closure modes, selection/configuration changes, empty windows and real Cairo
navigation. `gl_histogram_viewport.jl` checks actual native GL bounds and
upper/lower geometry with stopped, explicitly closed screens; run it separately
with an external deadline like the other native GL checks.

`histogram_bars` checks actual native Hist1D rectangles against independent exact
bin-edge arithmetic, single-bin areas, gap/dodge styles, theme width precedence,
labels, closure/normalization modes, zero-height bins and changing bin counts.
It also covers cached height/window updates, empty-window restoration, rejected
geometry and existing live-owner disposal. `gl_histogram_bars.jl` checks native
GL rectangle geometry and substantial filled pixel area for constant inputs;
run it separately with an external deadline and a native display. Screens are
stopped and explicitly closed. These checks do not assert pixel-exact areas.

`rng`, `task_rng`, and `hmc_rng` compare complete default RNG state and actual
future child streams, alongside explicit sampler RNGs and fixed-work results.
They cover BAT-owned listener/Timer/disposal enrollment, legitimate callback
randomness, failure/retry boundaries, and public HMC/MultiTrafo sampling.
`preinit_dispose` also checks the separate disposal controller's RNG.
The HMC and pre-initialization paired campaigns establish an ordinary Base
blocking-wait scheduling condition before their seeds/checkpoints and keep
plain/plain controls. Julia 1.13.2 can create a per-thread scheduler task during
its first empty-queue wait even without BAT; these comparisons do not claim
neutrality for that cold runtime initialization.
These Cairo checks do not establish native GL RNG neutrality.
`gl_rng.jl` separately checks actual cold/cached warmup, native and inline
display, screen reuse, callback draws/forks, startup failures and cleanup.
`gl_hmc_rng.jl` runs the same maintained HMC/MultiTrafo campaign with actual
native screens, both cold and cached, and checks renderer termination during
disposal. Run each in its own externally bounded process with a project
containing GLMakie and this checkout, on a working native display:

```sh
timeout 600 julia --startup-file=no --threads=2,1 --project=/path/to/gl-project test/optional/makie/gl_rng.jl
timeout 600 julia --startup-file=no --threads=2,1 --project=/path/to/gl-project test/optional/makie/gl_hmc_rng.jl
```

The native tests state their ordinary-wait runtime condition explicitly, retain
the plain/plain campaign control, and leave automatic Scene finalizers and
caller-owned direct Makie display outside their claims. They are separate from
headless CI and do not imply a platform or renderer-version compatibility matrix.
The optional MCMC owner runner provides native SIGINT coverage separately;
direct exception tests in `task_rng` are not native signal tests.

`hexbin` checks absolute weighted-count color admission after native aggregation
and threshold filtering. It covers independent BigFloat oracles, generated count
families, native geometry and colorbar parity, actual public figures/Apply,
SpecApi updates, fail-stop disposal and recreation, and bounded validation
allocations. `hexbin_lifecycle` demands the actual rejection inside BAT's grid
update and existing live view guard, checking failed/incomplete status, observer
release and independent-figure survival. Its focused owner fixture does not
claim sampler/RNG parity for an entire sampling run.

`gl_hexbin.jl` checks the same count adapter on native GL with actual Colorbars,
direct and SpecApi updates, rejected-figure disposal/recreation and survival of
an independent screen. Run it separately with an external deadline, as above.
It creates hidden stopped screens and closes only screens owned by the test.
The GL hexbin check also compares actual rendered pixels across admitted count
magnitudes, including the reciprocal-span boundary. This catches shader color
collapse that a finite CPU color mapping and nonempty image alone would miss.
