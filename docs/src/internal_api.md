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
BAT.apply_bounds_and_eval_posterior_logval_strict!
BAT.apply_bounds_and_eval_posterior_logval!
BAT.apply_bounds!
BAT.bat_sampler
BAT.bg_R_2sqr
BAT.find_localmodes
BAT.get_bin_centers
BAT.create_hypercube
BAT.create_hyperrectangle
BAT.distribution_logpdf
BAT.distribution_logpdf!
BAT.drop_low_weight_samples
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
BAT.integrate_hyperrectangle_cov
BAT.issymmetric_around_origin
BAT.log_volume
BAT.mcmc_startval!
BAT.modify_hypercube!
BAT.modify_integrationvolume!
BAT.proposal_rand!
BAT.reduced_volume_hm
BAT.spatialvolume
BAT.sum_first_dim
BAT.var_bounds
BAT.wgt_effective_sample_size
```
