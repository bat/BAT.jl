# Experimental features

These are experimental features. Forward/backward compatibility does *not*
follow [Julia's semantic versioning rules](https://julialang.github.io/Pkg.jl/v1/compatibility/).
Instead, compatibility is only guaranteed across changes in patch version, but
*not* across changes of minor (or major) version.

The features listed here are likely to transition to the stable API in future
versions, but may still evolve in a API-breaking fashion during that process.

## Affine-invariant ensemble sampling

`StretchMove` provides an experimental affine-invariant ensemble proposal:

```julia
algorithm = TransformedMCMC(proposal = StretchMove(), nwalkers = 32)
```

`nwalkers` is required explicitly and must be at least twice the transformed
dimension. The transformed initial ensemble must also have full affine rank.
One BAT chain is one coupled ensemble, and one sampler step is one complete
red-blue sweep over all walkers. Convergence diagnostics and effective sample
size pool only independent BAT ensembles, never walkers within an ensemble.

Only `RepetitionWeighting` is supported. Storing burn-in disables ensemble ESS
because compressed histories may no longer align by sweep. Affine invariance
does not solve multimodality, and acceptance rate alone does not demonstrate
convergence.

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
