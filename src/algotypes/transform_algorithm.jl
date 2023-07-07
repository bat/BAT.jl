# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractTransformTarget

Abstract type for measure transformation targets.
"""
abstract type AbstractTransformTarget end
export AbstractTransformTarget



"""
    abstract type AbstractTransformToUnitspace <: AbstractTransformTarget

Abstract type for measure transformation targets that specify a
transformation into the unit hypercube.
"""
abstract type AbstractTransformToUnitspace <: AbstractTransformTarget end
export AbstractTransformToUnitspace


"""
    abstract type AbstractTransformToInfinite <: AbstractTransformTarget

Abstract type for measure transformation targets that specify are
transformation into unbounded space.

The measure that results from such a transformation must be unbounded in all
dimensions and that should taper off to zero somewhat smoothly in any
direction.
"""
abstract type AbstractTransformToInfinite <: AbstractTransformTarget end
export AbstractTransformToInfinite


"""
    abstract type TransformAlgorithm

Abstract type for measure transformation algorithms.
"""
abstract type TransformAlgorithm end
export TransformAlgorithm


"""
    bat_transform(
        target::AbstractTransformTarget,
        measure::AnyMeasureLike,
        [algorithm::TransformAlgorithm]
    )

Transform `measure` to another variate space defined/implied by `target`.

Returns a NamedTuple of the shape

```julia
(result = new_measure::AnyMeasureLike, trafo = vartrafo::Function, ...)
```

Result properties not listed here are algorithm-specific and are not part
of the stable public API.

!!! note

    Do not add add methods to `bat_transform`, add methods to
    `bat_transform_impl` instead.
"""
function bat_transform end
export bat_transform

function bat_transform_impl end


function bat_transform(
    target::AbstractTransformTarget,
    measure::AnyMeasureLike,
    algorithm::TransformAlgorithm = bat_default_withinfo(bat_transform, Val(:algorithm), target, measure)
)
    r = bat_transform_impl(target, measure, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_transform), ::Val{:algorithm}, x::TransformAlgorithm)
    "Using transform algorithm $x"
end



"""
    struct DoNotTransform <: AbstractTransformTarget

The identity transformation target, specifies that densities
should not be transformed.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct DoNotTransform <: AbstractTransformTarget end
export DoNotTransform



"""
    struct IdentityTransformAlgorithm <: TransformAlgorithm

A no-op transform algorithm that leaves any measure unchanged.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct IdentityTransformAlgorithm <: TransformAlgorithm end
export IdentityTransformAlgorithm


function bat_transform_impl(::DoNotTransform, measure::AnyMeasureLike, algorithm::IdentityTransformAlgorithm)
    (result = convert(BATMeasure, measure), trafo = identity)
end


# ToDo: Merge PriorToUniform and PriorToGaussian into PriorTo{Uniform|Normal}.

"""
    struct PriorToUniform <: AbstractTransformToUnitspace

Specifies that posterior densities should be transformed in a way that makes
their pior equivalent to a uniform distribution over the unit hypercube.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct PriorToUniform <: AbstractTransformToUnitspace end
export PriorToUniform

_distribution_density_trafo(target::PriorToUniform, measure::DistMeasure) = DistributionTransform(Uniform, parent(measure))

function bat_transform_impl(target::PriorToUniform, measure::DistMeasure{<:StandardUniformDist}, algorithm::IdentityTransformAlgorithm)
    (result = measure, trafo = identity)
end


"""
    struct PriorToGaussian <: AbstractTransformToInfinite

Specifies that posterior densities should be transformed in a way that makes
their pior equivalent to a standard multivariate normal distribution with an
identity covariance matrix.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct PriorToGaussian <: AbstractTransformToInfinite end
export PriorToGaussian

_distribution_density_trafo(target::PriorToGaussian, measure::DistMeasure) = DistributionTransform(Normal, parent(measure))

function bat_transform_impl(target::PriorToGaussian, measure::DistMeasure{<:StandardNormalDist}, algorithm::IdentityTransformAlgorithm)
    (result = measure, trafo = identity)
end


_get_deep_transformable_base(measure::DistMeasure) = measure
_get_deep_transformable_base(measure::AbstractPosteriorMeasure) = _get_deep_transformable_base(getprior(measure))
# ToDo: Try via basemeasure as well:
_get_deep_transformable_base(measure::AbstractMeasure) = transport_origin(measure)


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


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::DistMeasure, algorithm::PriorSubstitution)
    trafo = _distribution_density_trafo(target, density)
    transformed_density = DistMeasure(trafo.target_dist)
    (result = transformed_density, trafo = trafo)
end


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::AbstractPosteriorMeasure, algorithm::PriorSubstitution)
    orig_prior = getprior(density)
    orig_likelihood = getlikelihood(density)
    new_prior, trafo = bat_transform_impl(target, orig_prior, algorithm)
    new_likelihood = Transformed(orig_likelihood, trafo, TDNoCorr())
    (result = PosteriorMeasure(new_likelihood, new_prior), trafo = trafo)
end


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::Renormalized, algorithm::PriorSubstitution)
    new_parent_density, trafo = bat_transform_impl(target, parent(density), algorithm)
    (result = Renormalized(new_parent_density, density.logrenormf), trafo = trafo)
end


# ToDo: Support bat_transform for vectors of variates and DensitySampleVector?


unshaping_trafo(::ArrayShape{Real, 1}) = identity
unshaping_trafo(vs::AbstractValueShape) = inverse(vs)


function transform_and_unshape(trafotarget::AbstractTransformTarget, measure::Any)
    transform_and_unshape(bat_transform, trafotarget, measure)
end

function transform_and_unshape(bat_trafofunc::Function, trafotarget::AbstractTransformTarget, measure::BATMeasure)
    trafoalg = bat_default(bat_trafofunc, Val(:algorithm), trafotarget, measure)
    transformed_density, initial_trafo = bat_trafofunc(trafotarget, measure, trafoalg)
    us_trafo = unshaping_trafo(varshape(transformed_density))
    result_density = us_trafo(transformed_density)
    result_trafo = us_trafo ∘ initial_trafo
    return result_density, result_trafo
end



"""
    bat_transform(
        f::Function,
        smpls::DensitySampleVector,
        [algorithm::TransformAlgorithm]
    )::DensitySampleVector
"""
function bat_transform(
    f::Function,
    smpls::DensitySampleVector,
    algorithm::TransformAlgorithm = bat_default_withinfo(bat_transform, Val(:algorithm), f, smpls)
)
    r = bat_transform_impl(f, smpls, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


struct SampleTransformation <: TransformAlgorithm end

function bat_transform_impl(f::Function, smpls::DensitySampleVector, ::SampleTransformation)
    (result = broadcast_arbitrary_trafo(f, smpls),)
end
