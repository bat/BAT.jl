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

`bounded_kde` compares supported KDEs with independent reflected Gaussian
mixtures, checks concentrated and narrow-support densities, actual probability
geometry, coordinate/weight scaling and explicit numerical/resource limits.
`kde_integration` exercises all four KDE recipes through static, ordinary and
live entry points, original-target support forwarding, explicit rendered edges,
transposition and empty/window/support transitions. The precompile workload uses
truthful bounded samples so strict support checks run during extension building.
