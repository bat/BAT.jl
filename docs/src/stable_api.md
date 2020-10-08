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
logvalof

bat_default
bat_eff_sample_size
bat_findmedian
bat_findmode
bat_initval
bat_integrate
bat_integrated_autocorr_len
bat_marginalmode
bat_read
bat_rng
bat_sample
bat_write

BAT.fft_autocor
BAT.fft_autocov
BAT.eval_logval_unchecked

AbstractDensity
AbstractMCMCWeightingScheme
AbstractPosteriorDensity
AbstractSampleGenerator
AdaptiveMHTuning
HamiltonianMC
AHMIntegration
AnyDensityLike
AnyIIDSampleable
AnySampleable
ARPWeighting
AutocorLenAlgorithm
BrooksGelmanConvergence
CholeskyPartialWhitening
CholeskyWhitening
DensitySample
DensitySampleVector
DistLikeDensity
ExplicitInit
GelmanRubinConvergence
GeyerAutocorLen
IIDSampling
InitFromIID
InitFromSamples
InitFromTarget
InitvalAlgorithm
IntegrationAlgorithm
LogDVal
MaxDensityLBFGS
MaxDensityNelderMead
MaxDensitySampleSearch
MCMCAlgorithm
MCMCBurninAlgorithm
MCMCChainPoolInit
MCMCConvergenceTest
MCMCInitAlgorithm
MCMCIterator
MCMCMultiCycleBurnin
MCMCNoOpTuning
MCMCSampling
MCMCTuningAlgorithm
MetropolisHastings
MHProposalDistTuning
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
