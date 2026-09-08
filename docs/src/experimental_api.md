# Experimental features

These are experimental features. Forward/backward compatibility does *not*
follow [Julia's semantic versioning rules](https://julialang.github.io/Pkg.jl/v1/compatibility/).
Instead, compatibility is only guaranteed across changes in patch version, but
*not* across changes of minor (or major) version.

The features listed here are likely to transition to the stable API in future
versions, but may still evolve in a API-breaking fashion during that process.

## Ensemble MCMC proposals

BAT provides three experimental proposals that update a coupled walker
ensemble in transformed coordinates:

```julia
algorithm = TransformedMCMC(proposal = StretchMove(), nwalkers = 32)
```

Only `StretchMove` and `DEMove` are affine-equivariant.

For an active walker ``x`` and walkers drawn from its frozen complement, the
proposal laws are:

- `StretchMove(scale = 2)` sets ``a = scale`` and draws ``z`` on ``[1/a, a]``
  with density proportional to ``z^{-1/2}``, proposes ``x' = y + z(x-y)``,
  and uses the Hastings factor ``z^{d-1}``.
- `DEMove(gamma0 = nothing, sigma = 1e-5)` proposes
  ``x' = x + gamma (y_1-y_2)`` from two distinct complement walkers, where
  ``gamma = gamma0 (1 + sigma epsilon)`` and ``epsilon`` is standard normal.
  The default `gamma0` is ``2.38 / sqrt(2d)`` after the transformed dimension
  is known.
- `DESnookerMove(scale = 1.7)` forms the unit direction ``u`` from a reference
  complement walker ``z`` to ``x`` and proposes
  ``x' = x + scale u (u' y_1 - u' y_2)``. Its Hastings factor is
  ``(||x'-z|| / ||x-z||)^{d-1}``; degenerate or non-finite directions are
  rejected without evaluating the target.

`nwalkers` is required explicitly. `StretchMove` needs at least ``2d``
walkers; `DEMove` and `DESnookerMove` need at least `max(2d, 4)`. The
transformed initial ensemble must contain only finite coordinates and have full
affine rank. One BAT chain is one coupled ensemble, and one sampler step is one
complete sweep over all walkers. Stretch and DE use two complementary groups;
the snooker move uses four. Convergence diagnostics and effective sample size
pool only independent BAT ensembles, never walkers within an ensemble.

`SequentialExec()` is the default executor. `MultiThreadedExec()` evaluates
the conditionally independent walkers within each active group concurrently;
groups themselves remain ordered so that each complement is frozen during its
update. Walker-keyed random streams make seeded results independent of thread
scheduling. Other executors, including `DistributedExec`, are rejected.

The moves may be components of a fixed-weight `MCMCMultiProposal`:

```julia
proposal = MCMCMultiProposal(
    proposals = BAT.MCMCProposal[StretchMove(), DEMove(), DESnookerMove()],
    picking_rule = [4, 2, 1],
)
algorithm = TransformedMCMC(proposal = proposal, nwalkers = 32)
```

An integer picking rule follows its deterministic weighted cycle; a
`Categorical` rule samples its fixed probabilities. The component is selected
once per sampler step, then the selected ensemble move completes its whole
sweep. Adaptive mixture-weight tuning is not supported for mixtures containing
an ensemble move.

Only `RepetitionWeighting` is supported. Storing burn-in disables ensemble ESS
because compressed histories may no longer align by sweep. Ensemble moves and
their mixtures require `NoAdaptiveTransform` and `NoMCMCTransformTuning`; the
individual moves also use `NoMCMCProposalTuning`. Affine invariance does not
solve multimodality, and neither acceptance rate nor a finite-run moment check
demonstrates convergence.

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
StretchMove
DEMove
DESnookerMove
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
```
