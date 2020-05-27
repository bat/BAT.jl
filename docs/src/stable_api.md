# API Documentation

This is the stable public API of BAT. Forward/backward compatibility follows
[Julia's semantic versioning rules](https://julialang.github.io/Pkg.jl/v1/compatibility/).


```@meta
DocTestSetup  = quote
    using BAT
end
```

## Types

```@index
Pages = ["stable_api.md"]
Order = [:type]
```

## Functions and macros

```@index
Pages = ["stable_api.md"]
Order = [:macro, :function]
```

# Documentation


```@docs
bat_findmedian
bat_findmode
bat_marginalmode
bat_integrate
bat_read
bat_rng
bat_sample
bat_write

BAT.default_mode_estimator
BAT.default_sampling_algorithm
BAT.density_logval
BAT.density_logvalgrad

AbstractDensity
AbstractMCMCTuningStrategy
AbstractPosteriorDensity
AbstractWeightingScheme
AdaptiveMetropolisTuning
AHMC
ARPWeighting
BrooksGelmanConvergence
DensitySample
DensitySampleVector
DistLikeDensity
GelmanRubinConvergence
LinDVal
LogDVal
MaxDensityLBFGS
MaxDensityNelderMead
MaxDensitySampleSearch
MCMCAlgorithm
MCMCBurninStrategy
MCMCInitStrategy
MCMCIterator
MetropolisHastings
ModeAsDefined
PosteriorDensity
RandResampling
RandSampling
RepetitionWeighting

BAT.AbstractModeEstimator
BAT.AbstractSamplingAlgorithm
BAT.AnyPosterior
BAT.GenericDensity

ValueShapes.totalndof
ValueShapes.varshape
```
