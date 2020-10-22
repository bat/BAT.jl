# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    AbstractDensityTransformTarget

Abstract type for probability density transformation targets.
"""
abstract type AbstractDensityTransformTarget end
export AbstractDensityTransformTarget



"""
    AbstractTransformToUnitspace <: AbstractDensityTransformTarget

Abstract type for density transformation targets what specify a
transformation into the unit hypercube.
"""
abstract type AbstractTransformToUnitspace <: AbstractDensityTransformTarget end
export AbstractTransformToUnitspace


"""
    AbstractTransformToInfinite <: AbstractDensityTransformTarget

Abstract type for density transformation targets that specity are
transformation into unbounded space.

The density that results from such a transformation must be unbounded in all
dimensions and that should taper off to zero somewhat smoothly in any
direction.
"""
abstract type AbstractTransformToInfinite <: AbstractDensityTransformTarget end
export AbstractTransformToInfinite


"""
    TransformAlgorithm

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

Returns a NamedTuple: (result = x::AbstractDensity, ...)

Result properties not listed here are algorithm-specific and are not part
of the stable BAT API.

!!! note

    Do not add add methods to `bat_transform`, add methods to
    `bat_transform_impl` instead (same arguments).
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
    "Using tranform algorithm $x"
end



"""
    NoDensityTransform <: AbstractDensityTransformTarget

The identity density transformation target, specifies that densities
should not be transformed.
"""
struct NoDensityTransform <: AbstractDensityTransformTarget end
export NoDensityTransform


"""
    DensityIdentityTransform <: TransformAlgorithm

A no-op density transform algorithm that leaves any density unchanged.
"""
struct DensityIdentityTransform <: AbstractDensityTransformTarget end
export DensityIdentityTransform


function bat_transform_impl(target::NoDensityTransform, density::AnyDensityLike, algorithm::DensityIdentityTransform)
    convert(AbstractDensity, density)
end



"""
    PriorToUniform <: AbstractTransformToUnitspace

Specifies that posterior densities should be transformed in a way that makes
their pior equivalent to a uniform distribution over the unit hypercube.
"""
struct PriorToUniform <: AbstractTransformToUnitspace end
export PriorToUniform

_prior_trafo(target::PriorToUniform, prior::DistributionDensity) = DistributionTransform(Uniform, parent(prior))



"""
    PriorToGaussian <: AbstractTransformToInfinite

Specifies that posterior densities should be transformed in a way that makes
their pior equivalent to a standard multivariate normal distribution with an
identity covariance matrix.
"""
struct PriorToGaussian <: AbstractTransformToInfinite end
export PriorToGaussian

_prior_trafo(target::PriorToGaussian, prior::DistributionDensity) = DistributionTransform(Normal, parent(prior))



"""
    FullDensityTransform <: TransformAlgorithm

Transform the density as a whole a given specified target space. Operations
that use the gradient of the density will require to the `log(abs(jacobian))`
of the transformation to be auto-differentiable.
"""
struct FullDensityTransform <: TransformAlgorithm end
export FullDensityTransform

function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::AbstractPosteriorDensity, algorithm::FullDensityTransform)
    orig_prior = getprior(density)
    trafo = _prior_trafo(target, orig_prior)
    (result = TransformedDensity(density, trafo, TDLADJCorr()),)
end



"""
    PriorSubstitution <: TransformAlgorithm

Substitute the prior by a given distribution and transform the
likelihood accordingly. The `log(abs(jacobian))` of the transformation does
not need to be auto-differentiable even for operations that use the
gradient of the posterior.
"""
struct PriorSubstitution <: TransformAlgorithm end
export PriorSubstitution


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::AbstractPosteriorDensity, algorithm::PriorSubstitution)
    orig_prior = getprior(density)
    orig_likelihood = getlikelihood(density)
    trafo = _prior_trafo(target, orig_prior)
    new_likelihood = TransformedDensity(orig_likelihood, trafo, TDNoCorr())
    new_prior = trafo.target_dist
    (result = PosteriorDensity(new_likelihood, new_prior),)
end
