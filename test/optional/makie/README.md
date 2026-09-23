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

`gl_smoke.jl` additionally tests a native GLMakie close and an unrelated screen.
Run it with an external process deadline in a scratch environment containing
this checkout and GLMakie, on a machine with a working display/OpenGL context.
It creates two hidden native windows and closes only those owned by the test.
This manual check is separate from the headless CairoMakie CI job.

The `static_result` case checks the public interactive figure return, explicit
display and deferred rendering, weighted histogram contents, working controls,
independent figures, and scoped themes. `precompile_workload` and
`precompile_failure` exercise the real workload and its owned-figure cleanup on
success and injected failure. These cases also run in separate bounded processes.
The precompile cases add narrow test-only methods at real workload boundaries;
keep them isolated rather than including them in a shared test process.

The `domain` case checks live bounds against analytic and pre-change oracles,
including native finite filtering and warmed allocation growth with fixed walker
topology. `domain_publication` checks those bounds through real ingestion and
flushes, including replacement of pending extremes and prior-generation ownership.

The `lifecycle_figure` case checks completed live figure retrieval through public
sampling, saving and controls, plus owning-thread checks, nonwaiting guards,
callback reentrancy, actual disposal, and focused lifecycle boundary fixtures.
