# Native MCMC owner interruption checks

These optional tests send actual SIGINT to isolated Julia processes. Python 3
enforces an external deadline for every process and retains logs, commands,
script hashes and signal times. A hung or crashed child fails; emergency process
termination never counts as successful MCMC cleanup. They are separate from the
ordinary Julia test process.

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

Both cases use the actual production vector/scalar iterator, with barriers only
inside fixture-owned numerical steps:

- `vector_owner` interrupts an owner whose two workers are blocked in a numerical
  step, then releases those steps. It checks cancellation, joining, flushing, and
  no later numerical writes.
- `vector_drain` keeps the workers blocked, interrupts the owner again during
  its join, and only then releases the workers externally. One worker then fails
  and the diagnostic logger also throws; the original interruption must still
  reach the caller after both workers terminate.

Tests assert child termination before any fixture cleanup. They do not replace
production owner methods or inject exceptions into running tasks. The ordinary
suite covers successful sampler/RNG parity, public phase/callback behavior,
child-failure aggregation, partial-prefix drainage, and timer closure. A native
signal in the enqueue-to-registry instruction gap would require a production
injection seam; this optional harness makes no such claim.

The production group's timer is local and has no public enumeration interface;
these native cases do not inspect that object or the original exception identity
and raw backtrace. The driver logs external release times relative to actual
SIGINT sends so the repeated-join interleaving remains auditable.
