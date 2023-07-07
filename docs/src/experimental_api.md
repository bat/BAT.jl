# Experimental features

These are experimental features. Forward/backward compatibility does *not*
follow [Julia's semantic versioning rules](https://julialang.github.io/Pkg.jl/v1/compatibility/).
Instead, compatibility is only guaranteed across changes in patch version, but
*not* across changes of minor (or major) version.

The features listed here are likely to transition to the stable API in future
versions, but may still evolve in a API-breaking fashion during that process.

```@docs
bat_compare
bat_marginalmode
BAT.default_context
BAT.DistributionTransform
BAT.enable_error_log
BAT.error_log
BAT.EvalException
BAT.ext_default
BAT.get_adselector
BAT.get_context
BAT.LogUniform
BAT.PackageExtension
BAT.pkgext
BAT.set_rng
BAT.AnyMeasureLike
BATContext
BridgeSampling
CuhreIntegration
DivonneIntegration
EllipsoidalNestedSampling
GridSampler
HierarchicalDistribution
PriorImportanceSampler
ReactiveNestedSampling
renormalize_density
SobolSampler
SuaveIntegration
truncate_density
ValueAndThreshold
VEGASIntegration
```
