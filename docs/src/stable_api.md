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
bat_default
bat_eff_sample_size
bat_findmedian
bat_findmode
bat_integrate
bat_integrated_autocorr_len
bat_marginalmode
bat_read
bat_rng
bat_sample
bat_write

logvalgradof
logvalof

BAT.fft_autocor
BAT.fft_autocov
BAT.logvalof_unchecked

AHMC
AHMIntegration
ARPWeighting
AbstractDensity
AbstractDensityValue
AbstractMCMCTuningStrategy
AbstractPosteriorDensity
AbstractWeightingScheme
AdaptiveMetropolisTuning
AnyDensityLike
AnyIIDSampleable
AnySampleable
AutocorLenAlgorithm
BrooksGelmanConvergence
CholeskyPartialWhitening
CholeskyWhitening
DensitySample
DensitySampleVector
DistLikeDensity
GelmanRubinConvergence
GeyerAutocorLen
IIDSampling
IntegrationAlgorithm
LinDVal
LogDVal
MCMCAlgorithm
MCMCBurninStrategy
MCMCInitStrategy
MCMCIterator
MaxDensityLBFGS
MaxDensityNelderMead
MaxDensitySampleSearch
MetropolisHastings
ModeAsDefined
NoWhitening
OrderedResampling
PosteriorDensity
RandResampling
RepetitionWeighting
SokalAutocorLen
StatisticalWhitening
WhiteningAlgorithm

BAT.AbstractModeEstimator
BAT.AbstractSamplingAlgorithm
BAT.GenericDensity

ValueShapes.totalndof
ValueShapes.varshape
```
