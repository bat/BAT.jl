BAT.jl v3.0.0 Release Notes
===========================

New features
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


Breaking changes
----------------

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
