# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractDensityTransformTarget

Abstract type for probability density transformation targets.
"""
abstract type AbstractDensityTransformTarget end
export AbstractDensityTransformTarget



"""
    abstract type AbstractTransformToUnitspace <: AbstractDensityTransformTarget

Abstract type for density transformation targets that specify a
transformation into the unit hypercube.
"""
abstract type AbstractTransformToUnitspace <: AbstractDensityTransformTarget end
export AbstractTransformToUnitspace


"""
    abstract type AbstractTransformToInfinite <: AbstractDensityTransformTarget

Abstract type for density transformation targets that specify are
transformation into unbounded space.

The density that results from such a transformation must be unbounded in all
dimensions and that should taper off to zero somewhat smoothly in any
direction.
"""
abstract type AbstractTransformToInfinite <: AbstractDensityTransformTarget end
export AbstractTransformToInfinite


"""
    abstract type TransformAlgorithm

Abstract type for density transformation algorithms.
"""
abstract type TransformAlgorithm end
export TransformAlgorithm


"""
    bat_transform(
        target::AbstractDensityTransformTarget,
        density::AnyDensityLike,
        [algorithm::TransformAlgorithm]
    )::AbstractDensity

Transform `density` to another variate space defined/implied by `target`.

Returns a NamedTuple of the shape

```julia
(result = newdensity::AbstractDensity, trafo = vartrafo::AbstractVariateTransform, ...)
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
    target::AbstractDensityTransformTarget,
    density::AnyDensityLike,
    algorithm::TransformAlgorithm = bat_default_withinfo(bat_transform, Val(:algorithm), target, density)
)
    r = bat_transform_impl(target, density, algorithm)
    result_with_args(r, (algorithm = algorithm,))
end


function argchoice_msg(::typeof(bat_transform), ::Val{:algorithm}, x::TransformAlgorithm)
    "Using transform algorithm $x"
end



"""
    struct NoDensityTransform <: AbstractDensityTransformTarget

The identity density transformation target, specifies that densities
should not be transformed.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct NoDensityTransform <: AbstractDensityTransformTarget end
export NoDensityTransform



"""
    struct DensityIdentityTransform <: TransformAlgorithm

A no-op density transform algorithm that leaves any density unchanged.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct DensityIdentityTransform <: TransformAlgorithm end
export DensityIdentityTransform


function bat_transform_impl(target::NoDensityTransform, density::AnyDensityLike, algorithm::DensityIdentityTransform)
    (result = convert(AbstractDensity, density), trafo = IdentityVT(varshape(density)))
end



"""
    struct PriorToUniform <: AbstractTransformToUnitspace

Specifies that posterior densities should be transformed in a way that makes
their pior equivalent to a uniform distribution over the unit hypercube.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct PriorToUniform <: AbstractTransformToUnitspace end
export PriorToUniform

_distribution_density_trafo(target::PriorToUniform, density::DistributionDensity) = DistributionTransform(Uniform, parent(density))

function bat_transform_impl(target::PriorToUniform, density::StandardUniformDensity, algorithm::DensityIdentityTransform)
    (result = density, trafo = IdentityVT(varshape(density)))
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

_distribution_density_trafo(target::PriorToGaussian, density::DistributionDensity) = DistributionTransform(Normal, parent(density))

function bat_transform_impl(target::PriorToGaussian, density::StandardNormalDensity, algorithm::DensityIdentityTransform)
    (result = density, trafo = IdentityVT(varshape(density)))
end


"""
    struct FullDensityTransform <: TransformAlgorithm

Transform the density as a whole a given specified target space. Operations
that use the gradient of the density will require to the `log(abs(jacobian))`
of the transformation to be auto-differentiable.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct FullDensityTransform <: TransformAlgorithm end
export FullDensityTransform


_get_deep_prior_for_trafo(density::AbstractPosteriorDensity) = _get_deep_prior_for_trafo(getprior(density))
_get_deep_prior_for_trafo(density::DistributionDensity) = density

function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::AbstractPosteriorDensity, algorithm::FullDensityTransform)
    orig_prior = _get_deep_prior_for_trafo(density)
    trafo = _distribution_density_trafo(target, orig_prior)
    (result = TransformedDensity(density, trafo, TDLADJCorr()), trafo = trafo)
end


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::DistributionDensity, algorithm::FullDensityTransform)
    trafo = _distribution_density_trafo(target, density)
    (result = TransformedDensity(density, trafo, TDLADJCorr()), trafo = trafo)
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


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::AbstractPosteriorDensity, algorithm::PriorSubstitution)
    orig_prior = getprior(density)
    orig_likelihood = getlikelihood(density)
    new_prior, trafo = bat_transform_impl(target, orig_prior, algorithm)
    new_likelihood = TransformedDensity(orig_likelihood, trafo, TDNoCorr())
    (result = PosteriorDensity(new_likelihood, new_prior), trafo = trafo)
end


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::DistributionDensity, algorithm::PriorSubstitution)
    trafo = _distribution_density_trafo(target, density)
    transformed_density = DistributionDensity(trafo.target_dist)
    (result = transformed_density, trafo = trafo)
end
