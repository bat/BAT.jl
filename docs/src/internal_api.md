# Internal API

!!! note

    This is the documentation of BAT's internal API. The internal API is
    fully accessible to users, but all aspects of it are subject to
    change without notice. Functionalities of the internal API that, over
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
BAT.BasicMvStatistics
BAT.DataSet
BAT.HMIData
BAT.HMISettings
BAT.IntegrationVolume
BAT.KDTreePartitioning
BAT.OnlineMvCov
BAT.OnlineMvMean
BAT.OnlineUvMean
BAT.OnlineUvVar
BAT.PointCloud
BAT.SearchResult
BAT.SpacePartTree
BAT.TruncatedDensity
BAT.WhiteningResult
BAT.MCMCSampleGenerator

BAT.density_logval_type
BAT.estimate_finite_bounds
BAT.argchoice_msg
BAT.bat_sampler
BAT.bg_R_2sqr
BAT.find_marginalmodes
BAT.get_bin_centers
BAT.create_hypercube
BAT.create_hyperrectangle
BAT.distribution_logpdf
BAT.distribution_logpdf!
BAT.drop_low_weight_samples
BAT.find_hypercube_centers
BAT.fromuhc
BAT.fromuhc!
BAT.fromui
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
BAT.renormalize_variate!
BAT.renormalize_variate_impl
BAT.repetition_to_weights
BAT.spatialvolume
BAT.sum_first_dim
BAT.truncate_density
BAT.truncate_dist_hard
BAT.trunc_logpdf_ratio
BAT.var_bounds
```
