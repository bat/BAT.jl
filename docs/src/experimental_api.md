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

## Live Makie session lifetime

Create a fresh `BATVisualizer(BATMakieVisualization(...))` for each sampling run.
Successful completion drains the final observations and retains an interactive
figure. Closing the live figure disables observation while sampling continues.
To release a retained figure, call `BAT.dispose_visualizer!(vis)` on the thread
that created it. This operation requests disposal without joining active view
callbacks; inspect `BAT.visualizer_status(vis).cleanup_pending` if needed.
Live GLMakie initialization must run on Julia's main thread.

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
