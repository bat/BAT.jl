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
BAT.DataSet
BAT.HMIData
BAT.HMISettings
BAT.IntegrationVolume
BAT.KDTreePartitioning
BAT.LogFuncDensity
BAT.MCMCSampleGenerator
BAT.OnlineMvCov
BAT.OnlineMvMean
BAT.OnlineUvMean
BAT.OnlineUvVar
BAT.PointCloud
BAT.SearchResult
BAT.SpacePartTree
BAT.StandardMvNormal
BAT.StandardMvUniform
BAT.StandardUvNormal
BAT.StandardUvUniform
BAT.TransformedDensity
BAT.TruncatedDensity
BAT.VariateTransform
BAT.WhiteningResult

BAT.argchoice_msg
BAT.bat_sampler
BAT.bg_R_2sqr
BAT.create_hypercube
BAT.create_hyperrectangle
BAT.density_logval_type
BAT.proposaldist_logpdf
BAT.drop_low_weight_samples
BAT.eval_logval
BAT.find_hypercube_centers
BAT.find_marginalmodes
BAT.fromuhc
BAT.fromuhc!
BAT.fromui
BAT.get_bin_centers
BAT.getlikelihood
BAT.getprior
BAT.gr_Rsqr
BAT.hm_init
BAT.hm_integrate!
BAT.hm_whiteningtransformation!
BAT.hyperrectangle_creationproccess!
BAT.integrate_hyperrectangle_cov
BAT.is_log_zero
BAT.issymmetric_around_origin
BAT.log_volume
BAT.log_zero_density
BAT.modify_hypercube!
BAT.modify_integrationvolume!
BAT.partition_space
BAT.proposal_rand!
BAT.reduced_volume_hm
BAT.repetition_to_weights
BAT.spatialvolume
BAT.sum_first_dim
BAT.trunc_logpdf_ratio
BAT.truncate_density
BAT.truncate_dist_hard
BAT.var_bounds
```
