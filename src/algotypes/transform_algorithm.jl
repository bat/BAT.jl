# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractTransformTarget

Abstract type for probability density transformation targets.
"""
abstract type AbstractTransformTarget end
export AbstractTransformTarget

AbstractTransformTarget(::Type{Vector}) = ToRealVector()
Base.convert(::Type{AbstractTransformTarget}, x) = AbstractTransformTarget(x)


"""
    abstract type TransformAlgorithm

Abstract type for density transformation algorithms.
"""
abstract type TransformAlgorithm end
export TransformAlgorithm


"""
    bat_transform(
        how::AbstractTransformTarget,
        object,
        [algorithm::TransformAlgorithm]
    )

    bat_transform(
        f,
        object,
        [algorithm::TransformAlgorithm]
    )

Transform `object` to another variate space defined/implied by `target`,
res. using the transformation function `f`.

Returns a NamedTuple of the shape

```julia
(result = newdensity, trafo = vartrafo::Function, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    As a convenience,
    
    ```julia
    flat_smpls, f_flatten = bat_transform(Vector, measure)
    flat_smpls, f_flatten = bat_transform(Vector, samples)
    ```
    
    can be used to flatten a the variate type of a measure (res. samples of a
    measure) to something like `Vector{<:Real}`.
"""
function bat_transform end
export bat_transform


_convert_trafo_how(trafo_how) = trafo_how
_convert_trafo_how(::Type{<:Vector}) = AbstractTransformTarget(Vector)

_convert_trafor_from(trafo_from) = trafo_from
_convert_trafor_from(d::Distribution) = batmeasure(d)


function bat_transform_impl end

function bat_transform(trafo_how, trafo_from, algorithm::TransformAlgorithm, context::BATContext)
    new_trafo_how = _convert_trafo_how(trafo_how)
    new_trafo_from = _convert_trafor_from(trafo_from)
    orig_context = deepcopy(context)
    r = bat_transform_impl(new_trafo_how, new_trafo_from, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end

bat_transform(trafo_how, trafo_from) = bat_transform(trafo_how, trafo_from, get_batcontext())

bat_transform(trafo_how, trafo_from, algorithm) = bat_transform(trafo_how, trafo_from, algorithm, get_batcontext())

function bat_transform(trafo_how, trafo_from, context::BATContext)
    new_trafo_how = _convert_trafo_how(trafo_how)
    new_trafo_from = _convert_trafor_from(trafo_from)
    algorithm = bat_default_withinfo(bat_transform, Val(:algorithm), new_trafo_how, new_trafo_from)
    bat_transform(new_trafo_how, new_trafo_from, algorithm, context)
end


function argchoice_msg(::typeof(bat_transform), ::Val{:algorithm}, x::TransformAlgorithm)
    "Using transform algorithm $x"
end



"""
    struct DoNotTransform <: AbstractTransformTarget

The identity density transformation target, specifies that densities
should not be transformed.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct DoNotTransform <: AbstractTransformTarget end
export DoNotTransform



"""
    struct IdentityTransformAlgorithm <: TransformAlgorithm

A no-op density transform algorithm that leaves any density unchanged.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct IdentityTransformAlgorithm <: TransformAlgorithm end
export IdentityTransformAlgorithm


function bat_transform_impl(::DoNotTransform, measure::MeasureLike, ::IdentityTransformAlgorithm, ::BATContext)
    (result = batmeasure(measure), trafo = identity)
end


"""
    struct ToRealVector <: AbstractTransformTarget

Specifies that the input should be transformed into a measure over the space
of real-valued flat vectors.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct ToRealVector <: AbstractTransformTarget end
export ToRealVector


# ToDo: Merge PriorToUniform and PriorToGaussian into PriorTo{Uniform|Normal}.

"""
    struct PriorToUniform <: AbstractTransformTarget

Specifies that posterior densities should be transformed in a way that makes
their pior equivalent to a uniform distribution over the unit hypercube.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct PriorToUniform <: AbstractTransformTarget end
export PriorToUniform

_distmeasure_trafo(target::PriorToUniform, density::BATDistMeasure) = DistributionTransform(Uniform, Distribution(density))

function bat_transform_impl(target::PriorToUniform, density::BATDistMeasure{<:StandardUniformDist}, algorithm::IdentityTransformAlgorithm, context::BATContext)
    (result = density, trafo = identity)
end


"""
    struct PriorToGaussian <: AbstractTransformTarget

Specifies that posterior densities should be transformed in a way that makes
their pior equivalent to a standard multivariate normal distribution with an
identity covariance matrix.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct PriorToGaussian <: AbstractTransformTarget end
export PriorToGaussian

_distmeasure_trafo(target::PriorToGaussian, density::BATDistMeasure) = DistributionTransform(Normal, Distribution(density))

function bat_transform_impl(target::PriorToGaussian, density::BATDistMeasure{<:StandardNormalDist}, algorithm::IdentityTransformAlgorithm, context::BATContext)
    (result = density, trafo = identity)
end


"""
    struct FullMeasureTransform <: TransformAlgorithm

*BAT-internal, not part of stable public API.*

Transform the density as a whole a given specified target space. Operations
that use the gradient of the density will require to the `log(abs(jacobian))`
of the transformation to be auto-differentiable.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct FullMeasureTransform <: TransformAlgorithm end


_get_deep_prior_for_trafo(m::BATDistMeasure) = m
_get_deep_prior_for_trafo(m::AbstractPosteriorMeasure) = _get_deep_prior_for_trafo(getprior(m))


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, m::AbstractPosteriorMeasure, algorithm::FullMeasureTransform, context::BATContext)
    orig_prior = _get_deep_prior_for_trafo(m)
    trafo = _distmeasure_trafo(target, orig_prior)
    (result = BATPushFwdMeasure(trafo, m, KeepRootMeasure()), trafo = trafo)
end


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, m::BATDistMeasure, algorithm::FullMeasureTransform, context::BATContext)
    trafo = _distmeasure_trafo(target, m)
    (result = BATPushFwdMeasure(trafo, m, KeepRootMeasure()), trafo = trafo)
end



"""
    struct PriorSubstitution <: TransformAlgorithm

Substitute the prior by a given distribution and transform the
likelihood accordingly. The `log(abs(jacobian))` of the transformation does
not need to be auto-differentiable even for operations that use the
gradient of the posterior.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct PriorSubstitution <: TransformAlgorithm end
export PriorSubstitution


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::BATDistMeasure, algorithm::PriorSubstitution, context::BATContext)
    trafo = _distmeasure_trafo(target, density)
    transformed_density = BATDistMeasure(trafo.target_dist)
    (result = transformed_density, trafo = trafo)
end


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::AbstractPosteriorMeasure, algorithm::PriorSubstitution, context::BATContext)
    orig_prior = getprior(density)
    orig_likelihood = getlikelihood(density)
    new_prior, trafo = bat_transform_impl(target, orig_prior, algorithm, context)
    new_likelihood = _precompose_density(orig_likelihood, inverse(trafo))
    (result = PosteriorMeasure(new_likelihood, new_prior), trafo = trafo)
end


# ToDo: Support bat_transform for vectors of variates and DensitySampleVector?


# ToDo: Remove transform_and_unshape and use `ToRealVector` instead of `DoNotTransform` in algorithms?
function transform_and_unshape(trafotarget::AbstractTransformTarget, object::Any, context::BATContext)
    orig_measure = batmeasure(object)
    trafoalg = bat_default(bat_transform, Val(:algorithm), trafotarget, orig_measure)
    transformed_measure, initial_trafo = bat_transform(trafotarget, orig_measure, trafoalg, context)
    result_measure, unshaping_trafo = bat_transform(ToRealVector(), transformed_measure, UnshapeTransformation(), context)
    result_trafo = fcomp(unshaping_trafo, initial_trafo)
    return result_measure, result_trafo
end



"""
    struct SampleTransformation <: TransformAlgorithm

*BAT-internal, not part of stable public API.*
"""
struct SampleTransformation <: TransformAlgorithm end

function bat_transform_impl(f::Function, smpls::DensitySampleVector, ::SampleTransformation, context::BATContext)
    (result = broadcast_arbitrary_trafo(f, smpls), trafo = f)
end

function bat_transform_impl(shp::AbstractValueShape, smpls::DensitySampleVector, ::SampleTransformation, context::BATContext)
    (result = shp.(smpls), trafo = shp)
end


"""
    struct UnshapeTransformation <: TransformAlgorithm

*BAT-internal, not part of stable public API.*
"""
struct UnshapeTransformation <: TransformAlgorithm end

function bat_transform_impl(::ToRealVector, obj::Union{BATMeasure,DensitySampleVector}, ::UnshapeTransformation, context::BATContext)
    trafo = Base.Fix2(unshaped, varshape(obj))
    trafoalg = bat_default(bat_transform, Val(:algorithm), trafo, obj)
    bat_transform_impl(trafo, obj, trafoalg, context)
end

function bat_transform_impl(::Base.Fix2{typeof(unshaped),<:ArrayShape{<:Real,1}}, m::BATMeasure, ::FullMeasureTransform, context::BATContext)
    (result = m, trafo = identity)
end

function bat_transform_impl(f::Base.Fix2{typeof(unshaped)}, m::BATMeasure, ::FullMeasureTransform, context::BATContext)
    shp = f.x
    (result = unshaped(m, shp), trafo = f)
end

function bat_transform_impl(f::Base.Fix2{typeof(unshaped)}, smpls::DensitySampleVector, ::SampleTransformation, context::BATContext)
    unshape_vs = f.x
    @argcheck elshape(smpls.v) <= unshape_vs
    (result = unshaped.(smpls), trafo = f)
end
