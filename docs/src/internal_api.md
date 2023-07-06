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
BAT.AbstractProposalDist
BAT.AbstractSampleGenerator
BAT.BasicMvStatistics
BAT.ENSAutoProposal
BAT.ENSBound
BAT.ENSEllipsoidBound
BAT.ENSMultiEllipsoidBound
BAT.ENSNoBounds
BAT.ENSProposal
BAT.ENSRandomWalk
BAT.ENSSlice
BAT.ENSUniformly
BAT.LFDensity
BAT.LFDensityWithGrad
BAT.LogDVal
BAT.MCMCSampleGenerator
BAT.OnlineMvCov
BAT.OnlineMvMean
BAT.OnlineUvMean
BAT.OnlineUvVar
BAT.Renormalized
BAT.StandardMvNormal
BAT.StandardMvUniform
BAT.StandardUvNormal
BAT.StandardUvUniform
BAT.Transformed

BAT.WrappedNonBATDensity

BAT.argchoice_msg
BAT.bat_sampler
BAT.bg_R_2sqr
BAT.checked_logdensityof
BAT.default_val_numtype
BAT.default_var_numtype
BAT.density_valtype
BAT.drop_low_weight_samples
BAT.find_marginalmodes
BAT.fromuhc
BAT.fromuhc!
BAT.fromui
BAT.get_bin_centers
BAT.getlikelihood
BAT.getprior
BAT.gr_Rsqr
BAT.is_log_zero
BAT.issymmetric_around_origin
BAT.log_volume
BAT.log_zero_density
BAT.logvalof
BAT.proposal_rand!
BAT.proposaldist_logpdf
BAT.repetition_to_weights
BAT.smallest_credible_intervals
BAT.spatialvolume
BAT.sum_first_dim
BAT.trunc_logpdf_ratio
BAT.truncate_dist_hard
BAT.var_bounds
```
