BAT.jl v3.0.0-DEV Release Notes
===============================

**Note: Subject to change before the BAT.jl v3.0.0 release**.


New features
------------

* Support for [DensityInterface](https://github.com/JuliaMath/DensityInterface.jl): BAT will now accept any object that implements the DensityInterface API (specifically `DensityInterface.densitykind` and `DensityInterface.logdensityof`) as likelihoods. In return, all BAT priors and posteriors support the DensityInterface API as well.

* Support for [InverseFunctions](https://github.com/JuliaMath/InverseFunctions.jl) and [ChangesOfVariables](https://github.com/JuliaMath/ChangesOfVariables.jl): Parameter transformations in BAT now implement the DensityInterface API. Any function that supports

    * `InverseFunctions.inverse`
    * `ChangesOfVariables.with_logabsdet_jacobian`
    * `output_shape = f(input_shape::ValueShapes.AbstractValueShape)::AbstractValueShape`

can now be used as a parameter transformation in BAT.


Breaking changes
----------------

* `AbstractVariateTransform` and the function `ladjof` have been removed, BAT parameter transformation do not need to have a specific supertype any longer (see above).

* Pending: [`LogDVal` will deprecated soon and removed before the BAT.jl v3.0.0 release. Instead of

  ```julia
  likelihood = let data = mydata
      function(v)
          log_likelihood_value = ...
          return LogDVal(log_likelihood_value)
      end
  end
  ```

  use the [DensityInterface](https://github.com/JuliaMath/DensityInterface.jl) API (see above):

  ```julia
  likelihood = let data = mydata
      logfuncdensity(function(v)
          log_likelihood_value = ...
          return log_likelihood_value
      end)
  end
  ```

  or

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

  This allows for defining likelihoods without depending on BAT. Avoid creating custom subtypes of `BAT.AbstractMeasureOrDensity`.

* New behavior of `ValueShapes.NamedTupleShape` and  `ValueShapes.NamedTupleDist`: Due to changes in [ValueShapes](https://github.com/oschulz/ValueShapes.jl) v0.10, `NamedTupleShape` and `NamedTupleDist` now either (by default) use `NamedTuple` or (optionally) `ValueShapes.ShapedAsNT`, but no longer a mix of them. As a result, the behavior of BAT has changed as well when using a `NamedTupleDist` as a prior. For example, `mode(samples).result` returns a `NamedTuple` now directly.

* Use the new function `bat_report` to generate a sampling output report instead of `show(BAT.SampledDensity(samples))`.


New experimental features
-------------------------

* `BATContext` and `BAT.default_context`
* `BAT.EvalException`
* `BAT.DistributionTransform`
* `BAT.enable_error_log`
* `BAT.error_log`
* `BAT.LogUniform`
* `BridgeSampling`
* `EllipsoidalNestedSampling`
* `ReactiveNestedSampling`
* `renormalize_density`
* `truncate_density`
