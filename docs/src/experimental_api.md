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
BAT.SimpleMCMCProposalState
BAT.TemperingState

bat_makie_plot
bat_theme
bat_theme_dark
BAT.BATVisBackend
BATMakieRecipe
BATMakieVisualization
BATVisualizer
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
