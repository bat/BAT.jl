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
BAT.AbstractMCMCCallback
BAT.AbstractProposalDist
BAT.BasicMvStatistics
BAT.DataSet
BAT.GenericDensity
BAT.HMIData
BAT.HMISettings
BAT.IntegrationVolume
BAT.MCMCCallbackWrapper
BAT.MCMCSpec
BAT.OnlineMvCov
BAT.OnlineMvMean
BAT.OnlineUvMean
BAT.OnlineUvVar
BAT.PointCloud
BAT.SearchResult
BAT.WhiteningResult


BAT.apply_bounds
BAT.apply_bounds!
BAT.apply_bounds_and_eval_posterior_logval!
BAT.apply_bounds_and_eval_posterior_logval_strict!
BAT.autocrl
BAT.bat_sampler
BAT.bg_R_2sqr
BAT.calculate_localmode
BAT.create_hypercube
BAT.create_hyperrectangle
BAT.distribution_logpdf
BAT.distribution_logpdf!
BAT.drop_low_weight_samples
BAT.effective_sample_size
BAT.eval_density_logval
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
BAT.initial_params!
BAT.issymmetric_around_origin
BAT.log_volume
BAT.modify_hypercube!
BAT.modify_integrationvolume!
BAT.param_bounds
BAT.params_shape
BAT.proposal_rand!
BAT.spatialvolume
BAT.sum_first_dim
BAT.wgt_effective_sample_size
```
