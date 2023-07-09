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

bat_convergence
bat_default
bat_eff_sample_size
bat_findmedian
bat_findmode
bat_initval
bat_integrate
bat_read
bat_report
bat_sample
bat_write
bat_transform

get_batcontext
set_batcontext

distbind
distprod
lbqintegral

AbstractMCMCWeightingScheme
AbstractPosteriorMeasure
AbstractTransformTarget
AdaptiveMHTuning
AssumeConvergence
AutocorLenAlgorithm
BATContext
BATHDF5IO
BATIOAlgorithm
BrooksGelmanConvergence
CuhreIntegration
DensitySample
DensitySampleVector
DivonneIntegration
DoNotTransform
EffSampleSizeAlgorithm
EffSampleSizeFromAC
EvaluatedMeasure
ExplicitInit
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
MaxDensitySearch
MCMCAlgorithm
MCMCBurninAlgorithm
MCMCChainPoolInit
MCMCInitAlgorithm
MCMCMultiCycleBurnin
MCMCNoOpTuning
MCMCSampling
MCMCTuningAlgorithm
MetropolisHastings
MHProposalDistTuning
ModeAsDefined
OptimAlg
OrderedResampling
PosteriorMeasure
PriorSubstitution
PriorToGaussian
PriorToUniform
RandResampling
RepetitionWeighting
SampleMedianEstimator
SokalAutocorLen
SuaveIntegration
TransformAlgorithm
VEGASIntegration
BinningAlgorithm
FixedNBins
FreedmanDiaconisBinning
RiceBinning
ScottBinning
SquareRootBinning
SturgesBinning
ToRealVector

BAT.AbstractMedianEstimator
BAT.AbstractModeEstimator
BAT.AbstractSamplingAlgorithm
BAT.ConvergenceTest
```
