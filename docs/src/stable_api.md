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
ladjof
trafoof

BAT.fft_autocor
BAT.fft_autocov
BAT.eval_logval_unchecked

AbstractDensity
AbstractDensityTransformTarget
AbstractMCMCWeightingScheme
AbstractPosteriorDensity
AbstractSampleGenerator
AbstractTransformedDensity
AbstractTransformToInfinite
AbstractTransformToUnitspace
AbstractVariateTransform
AdaptiveMHTuning
AHMIntegration
AnyDensityLike
AnyIIDSampleable
AnySampleable
ARPWeighting
AutocorLenAlgorithm
bat_transform
BrooksGelmanConvergence
CholeskyPartialWhitening
CholeskyWhitening
DensityIdentityTransform
DensitySample
DensitySampleVector
DistLikeDensity
ExplicitInit
FullDensityTransform
GelmanRubinConvergence
GeyerAutocorLen
HamiltonianMC
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
NoDensityTransform
NoWhitening
OrderedResampling
PosteriorDensity
PriorSubstitution
PriorToGaussian
PriorToUniform
RandResampling
RepetitionWeighting
SokalAutocorLen
StatisticalWhitening
TransformAlgorithm
WhiteningAlgorithm

BAT.AbstractModeEstimator
BAT.AbstractSamplingAlgorithm
BAT.GenericDensity

ValueShapes.totalndof
ValueShapes.varshape
```
