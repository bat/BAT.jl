# Internal API

!!! note

    This is the documentation of BAT's internal API. The internal API is
    fully accessible to users, but all aspects of it are subject to
    change without deprecation. Functionalities of the internal API that, over
    time, turn out to require user access (e.g. to support advanced use cases)
    will be evolved to gain a stable interface and then promoted to the public
    API.

```@meta
DocTestSetup  = quote
    using BAT
end
```

## Types

```@index
Pages = ["internal_api.md"]
Order = [:type]
```

## Functions and macros

```@index
Pages = ["internal_api.md"]
Order = [:macro, :function]
```

# Documentation

```@docs
BAT.AbstractSampleGenerator
BAT.AnySampleable
BAT.BasicMvStatistics
BAT.BATMeasure
BAT.BATPushFwdMeasure
BAT.BATPwrMeasure
BAT.BATWeightedMeasure
BAT.CholeskyPartialWhitening
BAT.CholeskyWhitening
BAT.DensitySampleMeasure
BAT.ENSAutoProposal
BAT.ENSBound
BAT.ENSEllipsoidBound
BAT.ENSMultiEllipsoidBound
BAT.ENSNoBounds
BAT.ENSProposal
BAT.ENSRandomWalk
BAT.ENSSlice
BAT.ENSUniformly
BAT.FullMeasureTransform
BAT.LFDensity
BAT.LFDensityWithGrad
BAT.LogDVal
BAT.MCMCIterator
BAT.MCMCSampleGenerator
BAT.MeasureLike
BAT.NoWhitening
BAT.OnlineMvCov
BAT.OnlineMvMean
BAT.OnlineUvMean
BAT.OnlineUvVar
BAT.SampleTransformation
BAT.StandardMvNormal
BAT.StandardMvUniform
BAT.StandardUvNormal
BAT.StandardUvUniform
BAT.StatisticalWhitening
BAT.UnshapeTransformation
BAT.WhiteningAlgorithm

BAT.logvalof
BAT.fft_autocor
BAT.fft_autocov
BAT.argchoice_msg
BAT.bg_R_2sqr
BAT.checked_logdensityof
BAT.drop_low_weight_samples
BAT.find_marginalmodes
BAT.get_bin_centers
BAT.getlikelihood
BAT.getprior
BAT.gr_Rsqr
BAT.is_log_zero
BAT.issymmetric_around_origin
BAT.log_zero_density
BAT.repetition_to_weights
BAT.smallest_credible_intervals
BAT.sum_first_dim
BAT.supports_rand
BAT.trunc_logpdf_ratio
BAT.truncate_dist_hard
BAT.measure_support
```
