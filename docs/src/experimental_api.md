# Experimental features

These are experimental features. Forward/backward compatibility does *not*
follow [Julia's semantic versioning rules](https://julialang.github.io/Pkg.jl/v1/compatibility/).
Instead, compatibility is only guaranteed across changes in patch version, but
*not* across changes of minor (or major) version changes.

The features listed here are likely to transition to the stable API in future
versions, but may still evolve in a API-breaking fashion during that process.

```@docs
bat_compare
bat_rng
bat_marginalmode
BAT.ExternalDensity
BAT.FunnelDistribution
BAT.GaussianShell
BAT.MultimodalCauchy
CuhreIntegration
DivonneIntegration
GridSampler
HierarchicalDistribution
PartitionedSampling
PriorImportanceSampler
ReactiveNestedSampling
SobolSampler
SuaveIntegration
VEGASIntegration
```
