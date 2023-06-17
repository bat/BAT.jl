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

bat_convergence
bat_default
bat_eff_sample_size
bat_findmedian
bat_findmode
bat_initval
bat_integrate
bat_integrated_autocorr_len
bat_read
bat_report
bat_sample
bat_write
trafoof

BAT.bat_report!
BAT.eval_logval_unchecked
BAT.fft_autocor
BAT.fft_autocov

BAT.AbstractMeasureOrDensity

AbstractMCMCWeightingScheme
AbstractPosteriorMeasure
AbstractTransformed
AbstractTransformTarget
AbstractTransformToInfinite
AbstractTransformToUnitspace
AdaptiveMHTuning
AnyIIDSampleable
AnyMeasureOrDensity
AnySampleable
ARPWeighting
AssumeConvergence
AutocorLenAlgorithm
bat_transform
BATHDF5IO
BATIOAlgorithm
BrooksGelmanConvergence
CholeskyPartialWhitening
CholeskyWhitening
DensitySample
DensitySampleVector
DistLikeMeasure
DoNotTransform
EffSampleSizeAlgorithm
EffSampleSizeFromAC
ExplicitInit
FullMeasureTransform
GelmanRubinConvergence
GeyerAutocorLen
HamiltonianMC
IdentityTransformAlgorithm
IIDSampling
InitFromIID
InitFromSamples
InitFromTarget
InitvalAlgorithm
IntegrationAlgorithm
KishESS
LBFGSOpt
LogDVal
MaxDensitySearch
MCMCAlgorithm
MCMCBurninAlgorithm
MCMCChainPoolInit
MCMCInitAlgorithm
MCMCIterator
MCMCMultiCycleBurnin
MCMCNoOpTuning
MCMCSampling
MCMCTuningAlgorithm
MetropolisHastings
MHProposalDistTuning
ModeAsDefined
NelderMeadOpt
NoWhitening
OrderedResampling
PosteriorMeasure
PriorSubstitution
PriorToGaussian
PriorToUniform
RandResampling
RepetitionWeighting
SampledMeasure
SokalAutocorLen
StatisticalWhitening
TransformAlgorithm
WhiteningAlgorithm

BAT.AbstractModeEstimator
BAT.AbstractSamplingAlgorithm
BAT.ConvergenceTest
BAT.GenericDensity

ValueShapes.totalndof
ValueShapes.varshape
```
