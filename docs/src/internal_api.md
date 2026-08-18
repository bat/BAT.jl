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
BAT.BasicMvStatistics
BAT.BATMeasure
BAT.BATPushFwdMeasure
BAT.BATPwrMeasure
BAT.BATSuperpositionMeasure
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
BAT.JointLikelihood
BAT.LFDensity
BAT.AbstractTransformInit
BAT.LFDensityWithGrad
BAT.LogDVal
BAT.MCMCSampleGenerator
BAT.MeasureLike
BAT.NoWhitening
BAT.OnlineMvCov
BAT.OnlineMvMean
BAT.OnlineUvMean
BAT.OnlineUvVar
BAT.PathfinderTransformInit
BAT.PriorApproxTransformInit
BAT.SampleTransformation
BAT.StanLikeTuning
BAT.StandardMvNormal
BAT.StandardMvUniform
BAT.StandardUvNormal
BAT.StandardUvUniform
BAT.StatisticalWhitening
BAT.StepSizeAdaptor
BAT.DiagonalAffineTransform
BAT.LowRankAffineTransform
BAT.TriangularAffineTransform
BAT.UnshapeTransformation
BAT.WhiteningAlgorithm

BAT.argchoice_msg
BAT.bg_R_2sqr
BAT.checked_logdensityof
BAT.dist_samples_mean_zscores
BAT.drop_low_weight_samples
BAT.fft_autocor
BAT.fft_autocov
BAT.find_marginalmodes
BAT.get_bin_centers
BAT.get_iid_sampleable_approx
BAT.getlikelihood
BAT.getprior
BAT.gr_Rsqr
BAT.has_uhc_support
BAT.hmc_find_good_stepsize
BAT.hmc_nuts_transition
BAT.is_log_zero
BAT.issymmetric_around_origin
BAT.log_zero_density
BAT.logvalof
BAT.maximize_density
BAT.BispacedMeasure
BAT.pathfinder_gaussian_fit
BAT.repetition_to_weights
BAT.smallest_credible_intervals
BAT.sum_first_dim
BAT.supports_rand
BAT.transform_function
BAT.trunc_logpdf_ratio
BAT.truncate_dist_hard
```
