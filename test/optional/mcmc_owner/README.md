# Native MCMC owner interruption checks

These optional tests send actual SIGINT to isolated Julia processes. Python 3
enforces an external deadline for every process and retains logs, commands,
script hashes and signal times. A hung or crashed child fails; emergency process
termination never counts as successful MCMC cleanup. They are deliberately
separate from the ordinary Julia test process.

From the repository root:

```sh
julia --project=test/optional/mcmc_owner -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
python3 test/optional/mcmc_owner/runtests.py
```

Use `--julia /path/to/julia`, `--project /path/to/an/instantiated/environment`,
`--threads=1,1` or `--threads=2,1`, and `--timeout=300` as needed. One interactive
thread keeps the sampling owner on thread 1, where Julia delivers native SIGINT;
the numerical tasks use the unchanged default thread pool. Explicit development
of this checkout selects the intended BAT source on Julia 1.10 too. Each BAT log
prints `pathof(BAT)` for verification. POSIX process groups are required;
unsupported platforms exit 77 explicitly without claiming coverage.

Every run first executes a minimal BAT-free SIGINT control. If that control
crashes, times out or fails its assertion, the runner exits 2 and records the BAT
cases as unrun. No Julia version is automatically skipped or marked as an
expected failure. `--control-only` runs just this environment check.

The cases use the actual production vector/scalar iterator and public
`bat_sample`:

- `vector_owner` interrupts an owner whose two workers are blocked in an
  observation hook; checks cancellation, joining and no later numerical steps.
- `vector_stop` interrupts again before the stop hook releases those workers;
  requires a completed retry, not merely an attempted notification.
- `vector_drain` keeps the workers blocked after notification, interrupts the
  owner again during its join, and only then releases the workers externally.
- `public_owner` interrupts ordinary public sampling and checks that failed
  visualization finalization observes every worker already terminal.
- `init_pool_first`, `init_pool_second`, and `init_retry` interrupt the public
  sampling owner while two metadata writers are held at the selected real
  initialization boundary. They release those writers after a controlled delay
  and require cleanup to follow both joins, with no later metadata repopulation
  or callbacks. These groups intentionally have no participating stop hook.

All cases assert the original exception object and raw backtrace survive the
real vector boundary. The test backend observes the active owner catch through
its dispatched stop hook; it does not replace production owner methods or
inject exceptions into running tasks. Tests assert child termination before
any test-owned cleanup. Partial-prefix drainage and zero/partial enrollment
remain covered by the maintained core ownership tests: a deterministic native
signal in the enqueue-to-registry instruction gap would require a production
injection seam. This optional harness makes no such claim.

The production group's timer is local and has no public enumeration interface;
these native cases do not inspect that object. Timer ownership is assessed from
source review of the production normal/unwind close paths; this harness makes
no independent native assertion about the group's timer object.
The driver logs the external release times relative to actual SIGINT sends so
the blocked-writer and repeated-drain interleavings remain auditable.

The separately selected `makie_rng` case requires an instantiated Makie test
environment; it is not part of the default core-only cases:

```sh
python3 test/optional/mcmc_owner/runtests.py makie_rng --project=test/optional/makie
```

This is explicitly a **scheduler-initialized** native gate. Before the
enrollment checkpoint or child-stream measurements, it records full RNG state
around a real callback-free Base wait, checks ordinary draws, and requires a
second wait to preserve full state. It does not rewind state across either
wait or require the first wait to be RNG-neutral. Julia 1.13 can lazily create a
per-thread scheduler Task on an idle wait; the task first reaching that path
pays its split. Consequently, cold whole-operation neutrality is not claimed.
The recorded first-wait effect remains visible in the test log.

It then starts BAT's real callback Timer and listener, interrupts the owner during
the real `finish_visualizer!` wait on a fixture-owned listener barrier, and
then creates the real disposal coordinator from that interrupted owner. The
driver releases the listener only after the owner has inspected that
coordinator. It checks complete caller RNG state, subsequent real child
streams, persistent private launch-state progression, repeated-disposal
no-ops, explicit task termination, timer closure and history release before
fallback cleanup. The release-controller task is established before the RNG
checkpoint. This case does not construct a renderer or certify native GL RNG
neutrality. Its genuine SIGINT targets a stable wait; it does not claim delivery
inside the short enrollment instruction interval or inspect Base's hidden
Timer task. Existing constructor/SIGINT scopes are not enlarged for the test.

On the review host, glibc 2.44 with Julia 1.12.7 fails even the minimal control.
This matches [Julia issue #62683](https://github.com/JuliaLang/julia/issues/62683):
older signal delivery decodes glibc jump buffers using an obsolete pointer-guard
location. [The upstream fix](https://github.com/JuliaLang/julia/commit/f3f8cea305a93f14209c4d877eba23b25da6829a)
derives that representation at runtime. A passing test on another Julia build
does not erase the failing 1.12 installation result. Do not patch the runtime,
change application signal policy outside the test child, or disable signal
handling to make this gate pass.
