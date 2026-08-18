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

bat_bgml
bat_convergence
bat_default
bat_eff_sample_size
bat_findmedian
bat_findmode
bat_initval
bat_integrate
bat_read
bat_sample
bat_write
bat_transform

evalmeasure

empiricalof
samplesof
approxof
samplegenof
getess
evalinfo

get_batcontext
set_batcontext
log_batdebug

distbind
distprod
joint_likelihood
lbqintegral

AbstractMCMCWeightingScheme
AbstractPosteriorMeasure
TransformIntent
AdaptiveAffineTuning
AdaptiveMultiPropTuning
AssumeConvergence
AutocorLenAlgorithm
BATContext
BATHDF5IO
BATIOAlgorithm
BinningAlgorithm
BrooksGelmanConvergence
CuhreIntegration
DensitySample
DensitySampleVector
DivonneIntegration
DoNotTransform
DriftCommitSchedule
EffSampleSizeAlgorithm
EffSampleSizeFromAC
EvaluatedMeasure
ExplicitInit
FisherTransformTuning
FixedMGVISchedule
FixedNBins
FreedmanDiaconisBinning
GelmanRubinConvergence
GeyerAutocorLen
HamiltonianMC
IdentityTransformAlgorithm
MCMCGlobalProposal
IIDSampling
InitFromIID
InitFromSamples
InitFromTarget
InitvalAlgorithm
IntegrationAlgorithm
KishESS
MALAProposal
MaxDensitySearch
MCMCAlgorithm
MCMCBurninAlgorithm
MCMCChainPoolInit
MCMCRetryInit
MCMCInitAlgorithm
MCMCMultiCycleBurnin
MCMCMultiProposal
MCMCProposalTuning
MCMCTransformTuning
MGVISampling
ModeAsDefined
MultiProposalTuning
MultiTrafoTuning
NoMCMCProposalTuning
NoMCMCTransformTuning
OptimAlg
OptimizationAlg
OrderedResampling
PosteriorMeasure
PriorSubstitution
NormalBased
UniformBased
RAMTuning
RandomWalk
RandResampling
RepetitionWeighting
RiceBinning
SampleMedianEstimator
ScottBinning
SokalAutocorLen
SquareRootBinning
SturgesBinning
SuaveIntegration
ToRealVector
TransformAlgorithm
TransformedMaxDensity
TransformedMCMC
VEGASIntegration

BAT.unevaluated

BAT.AbstractMedianEstimator
BAT.AbstractModeEstimator
BAT.AbstractSamplingAlgorithm
BAT.ConvergenceTest
BAT.MGVISchedule
```
