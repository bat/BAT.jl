BAT.jl Release Notes
====================

BAT.jl v5.0.0
-------------

### Breaking changes

* `bat_sample`, `bat_integrate`, and `bat_findmode` still return their primary
  result in `.result`, but no longer add `.evaluated` or duplicate samples,
  masses, and modes to their result tuples. Use `evalmeasure` and the
  `samplesof`, `massof`, `mode`, `modes`, and `evalinfo` accessors when those
  results are needed together.

* `EvaluatedMeasure` now accepts samples through the `empirical` keyword
  instead of `samples`. Use the same accessors instead of its old result
  fields.

* For mode finding, `OptimAlg` and `OptimizationAlg` now select only the
  optimization backend. `pretransform` and `init` have moved to
  `TransformedMaxDensity`.

* `AbstractTransformTarget` has become `TransformIntent`. `PriorToNormal` and
  `PriorToUniform` have become `NormalBased` and `UniformBased`; deprecated
  aliases keep existing calls working for now.

* `OrderedResampling` has become `SystematicResampling`. A deprecated alias
  keeps existing calls working for now.

* The AdvancedHMC backend and dependency have been removed. `HamiltonianMC`
  now uses BAT's native multinomial NUTS implementation, so AdvancedHMC metric,
  integrator, termination, and tuning options no longer apply.

* `MALAProposal` now defaults to normal innovations instead of `TDist(1.0)`,
  uses the exact proposal-density correction, and tunes its step size.
  Gradient proposals now default to `FisherTransformTuning` instead of
  `RAMTuning`.

* Weighted `DensitySampleVector` statistics now treat weights as probability
  masses instead of repetition counts. Weighted quantiles use the inverse
  empirical CDF, and autocorrelation ESS uses recorded MCMC process order when
  available.

* `set_batcontext` now changes the process-wide default instead of task-local
  storage. Use `ScopedSettings.with(default_batcontext => context)` for a
  temporary context inherited by child tasks.

### New features

* The new `evalmeasure` interface returns an `EvaluatedMeasure` containing all
  available results from an algorithm. The existing `bat_sample`,
  `bat_integrate`, and `bat_findmode` APIs still work and keep `.result`.

* `EvaluatedMeasure` and `DensitySampleMeasure` provide first-class measures
  for algorithm results and empirical samples, with access through
  `samplesof`, `massof`, `mode`, `modes`, `getess`, and `evalinfo`.

* `HamiltonianMC` provides native multinomial NUTS. `MALAProposal` provides an
  exact Hastings correction and automatic step-size tuning.

* Gradient-based MCMC now defaults to affine transform tuning with
  `FisherTransformTuning` and its `DriftCommitSchedule`.

* `AdaptiveTransformChain` can tune composed transformations.

* `default_batcontext` supports scoped context overrides inherited by child
  tasks.

BAT.jl v4.0.0
-------------

### Breaking changes

Several algorithms have changed their names, but also their role:

* `MCMCSampling` has become `TransformedMCMC`.

* `MetropolisHastings` has become `RandomWalk`. It's parameters have
    changed (no deprecation for the parameter changes). Tuning and
    sample weighting scheme selection have moved to `TransformedMCMC`.

* `PriorToGaussian` has become `PriorToNormal`.

Partial deprecations are available for the above, so old code should
run more or less unchanged (with deprecation warnings). Also:

* `AdaptiveMHTuning` has become `AdaptiveAffineTuning`, but is now
  used as a parameter for `TransformedMCMC` (formerly `MCMCSampling`)
  instead of `RandomWalk` (formerly `MetropolisHastings`).

* `MCMCNoOpTuning` has become `NoMCMCTransformTuning`.

* The arguments of `HamiltonianMC` have changed.

* `MCMCTuningAlgorithm` has been replaced by `MCMCTransformTuning`.

* The `trafo` parameter of algorithms has been renamed to `pretransform`, the
  `trafo` field in algorithm results has been renamed to `f_pretransform`.

* `bat_report` has been deprecated in favor of `LazyReports.lazyreport`
  (drop-in compatible).


### New features

* Sampling, integration and mode-finding algorithms now generate a return
  value `result = ..., evaluated::EvaluatedMeasure = ..., ...)` if their
  target is a probability measure/distribution.

* The new `RAMTuning` is now the default (transform) tuning algorithm for
  `RandomWalk` (formerly `MetropolisHastings`). It typically results in a much
  faster burn-in process than `AdaptiveAffineTuning` (formerly
  `AdaptiveMHTuning`, the previous default).

* MCMC Sampling handles parameter scale and correlation adaptivity via
  via tunable space transformations instead of tuning covariance matrices
  in proposal distributions.
  
* MCMC tuning has been split into proposal tuning (algorithms of type
  `MCMCProposalTuning`) and transform turning (algorithms of type
  `MCMCTransformTuning`). Proposal tuning has now a much more limited role
  and often may be `NoMCMCProposalTuning()` (e.g. for `RandomWalk`).

* Added `MGVISampling` for Metric Gaussian Variational Inference.


BAT.jl v3.0.0
-------------

### Breaking changes

* `AbstractVariateTransform` and the function `ladjof` have been removed, BAT parameter transformations do not need to have a specific supertype any longer (see above).

* The new `BATContext` replaces passing `rng::AbstractRNG` random number generators around.

* Pending: `LogDVal` has been deprecated soon and will removed in BAT v3.1 or v3.2. Do *not* do this any longer:

  ```julia
  likelihood = let data = mydata
      function(v)
          log_likelihood_value = ...
          return LogDVal(log_likelihood_value)
      end
  end
  ```

  Instead, use the [DensityInterface](https://github.com/JuliaMath/DensityInterface.jl) API (see above), like this:

  ```julia
  likelihood = let data = mydata
      logfuncdensity(function(v)
          log_likelihood_value = ...
          return log_likelihood_value
      end)
  end
  ```

  or like this

  ```julia
  struct MyLikeLihood{D}
      data::D
  end

  @inline DensityInterface.DensityKind(::MyLikeLihood) = IsDensity()

  function DensityInterface.logdensityof(likelihood::MyLikeLihood, v)
      log_likelihood_value = ...
      return log_likelihood_value
  end

  likelihood = MyLikeLihood(mydata)
  ```

  This allows for defining likelihoods without depending on BAT.

* New behavior of `ValueShapes.NamedTupleShape` and  `ValueShapes.NamedTupleDist`: Due to changes in [ValueShapes](https://github.com/oschulz/ValueShapes.jl) v0.10, `NamedTupleShape` and `NamedTupleDist` now either (by default) use `NamedTuple` or (optionally) `ValueShapes.ShapedAsNT`, but no longer a mix of them. As a result, the behavior of BAT has changed as well when using a `NamedTupleDist` as a prior. For example, `mode(samples).result` returns a `NamedTuple` now directly.

* `SampledMeasure` (formerly `SampledDensity`) have been replaced by `EvaluatedMeasure`.

* Some type-pirating prior plotting recipes have been removed, to be re-added in a clean way.

* Pending: BAT will rely less on ValueShapes in the future. Do not use ValueShapes functionality directly where avoidable. Use `distprod` instead of using `ValueShapes.NamedTupleDist` directly, and favor using `bat_transform` instead of shaping and unshaping data using values shapes directly, if possible.

* Use the new function `bat_report` to generate a sampling output report instead of `show(BAT.SampledDensity(samples))`.

* The field types of `EvaluatedMeasure` have changed.


### New features
------------

* Support for [DensityInterface](https://github.com/JuliaMath/DensityInterface.jl): BAT will now accept any object that implements the DensityInterface API (specifically `DensityInterface.densitykind` and `DensityInterface.logdensityof`) as likelihoods. In return, all BAT priors and posteriors support the DensityInterface API as well.

* Support for [InverseFunctions](https://github.com/JuliaMath/InverseFunctions.jl) and [ChangesOfVariables](https://github.com/JuliaMath/ChangesOfVariables.jl): Parameter transformations in BAT now implement the DensityInterface API. Any function that supports

    * `InverseFunctions.inverse`
    * `ChangesOfVariables.with_logabsdet_jacobian`
    * `output_shape = f(input_shape::ValueShapes.AbstractValueShape)::AbstractValueShape`

can now be used as a parameter transformation in BAT.

* `BATContext`, `get_batcontext` and `set_batcontext`

* `bat_transform` with enhanced capabilities

* `distprod`, `distbind` and `lbqintegral` are the new ways to express priors and posteriors in BAT.

* `bat_report`

* `BAT.enable_error_log` (experimental)

* `BAT.error_log` (experimental)

* `BridgeSampling` (experimental)

* `EllipsoidalNestedSampling` (experimental)

* `ReactiveNestedSampling` (experimental)
