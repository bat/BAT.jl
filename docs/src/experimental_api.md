# Experimental features

These are experimental features. Forward/backward compatibility does *not*
follow [Julia's semantic versioning rules](https://julialang.github.io/Pkg.jl/v1/compatibility/).
Instead, compatibility is only guaranteed across changes in patch version, but
*not* across changes of minor (or major) version.

The features listed here are likely to transition to the stable API in future
versions, but may still evolve in a API-breaking fashion during that process.

```@docs
ARPWeighting
bat_compare
bat_integrated_autocorr_len
bat_marginalmode
BAT.auto_renormalize
BAT.batsampleable
BAT.BinnedModeEstimator
BAT.DistributionTransform
BAT.enable_error_log
BAT.error_log
BAT.EvalException
BAT.ext_default
BAT.get_adselector
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
WHAMISSampling

BAT.MCMCChainState
BAT.MCMCChainStateInfo
BAT.MCMCIterator
BAT.MCMCProposal
BAT.MCMCProposalState
BAT.MCMCProposalTunerState
BAT.MCMCState
BAT.MCMCTempering
BAT.MCMCTransformTunerState
BAT.PolarShellDistribution
BAT.TemperingState
```
