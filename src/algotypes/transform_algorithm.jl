# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type AbstractTransformTarget

Abstract type for probability density transformation targets.
"""
abstract type AbstractTransformTarget end
export AbstractTransformTarget


"""
    abstract type TransformAlgorithm

Abstract type for density transformation algorithms.
"""
abstract type TransformAlgorithm end
export TransformAlgorithm


"""
    bat_transform(
        target::AbstractTransformTarget,
        density::AnyMeasureOrDensity,
        [algorithm::TransformAlgorithm]
    )::AbstractMeasureOrDensity

Transform `density` to another variate space defined/implied by `target`.

Returns a NamedTuple of the shape

```julia
(result = newdensity::AbstractMeasureOrDensity, trafo = vartrafo::Function, ...)
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

bat_transform(trafo_how, trafo_from) = bat_transform(trafo_how, trafo_from, get_batcontext())

bat_transform(trafo_how, trafo_from, algorithm) = bat_transform(trafo_how, trafo_from, algorithm, get_batcontext())

function bat_transform(trafo_how, trafo_from, context::BATContext)
    algorithm = bat_default_withinfo(bat_transform, Val(:algorithm), trafo_how, trafo_from)
    bat_transform(trafo_how, trafo_from, algorithm, context)
end


function bat_transform(target::AbstractTransformTarget, density::AnyMeasureOrDensity, algorithm::TransformAlgorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_transform_impl(target, density, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
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


function bat_transform_impl(target::DoNotTransform, density::AnyMeasureOrDensity, algorithm::IdentityTransformAlgorithm, context::BATContext)
    (result = convert(AbstractMeasureOrDensity, density), trafo = identity)
end


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

_distribution_density_trafo(target::PriorToUniform, density::DistMeasure) = DistributionTransform(Uniform, parent(density))

function bat_transform_impl(target::PriorToUniform, density::DistMeasure{<:StandardUniformDist}, algorithm::IdentityTransformAlgorithm, context::BATContext)
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

_distribution_density_trafo(target::PriorToGaussian, density::DistMeasure) = DistributionTransform(Normal, parent(density))

function bat_transform_impl(target::PriorToGaussian, density::DistMeasure{<:StandardNormalDist}, algorithm::IdentityTransformAlgorithm, context::BATContext)
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


_get_deep_prior_for_trafo(density::DistMeasure) = density
_get_deep_prior_for_trafo(density::AbstractPosteriorMeasure) = _get_deep_prior_for_trafo(getprior(density))
_get_deep_prior_for_trafo(density::Renormalized) = _get_deep_prior_for_trafo(parent(density))


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::AbstractPosteriorMeasure, algorithm::FullMeasureTransform, context::BATContext)
    orig_prior = _get_deep_prior_for_trafo(density)
    trafo = _distribution_density_trafo(target, orig_prior)
    (result = Transformed(density, trafo, TDLADJCorr()), trafo = trafo)
end


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::DistMeasure, algorithm::FullMeasureTransform, context::BATContext)
    trafo = _distribution_density_trafo(target, density)
    (result = Transformed(density, trafo, TDLADJCorr()), trafo = trafo)
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


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::DistMeasure, algorithm::PriorSubstitution, context::BATContext)
    trafo = _distribution_density_trafo(target, density)
    transformed_density = DistMeasure(trafo.target_dist)
    (result = transformed_density, trafo = trafo)
end


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::AbstractPosteriorMeasure, algorithm::PriorSubstitution, context::BATContext)
    orig_prior = getprior(density)
    orig_likelihood = getlikelihood(density)
    new_prior, trafo = bat_transform_impl(target, orig_prior, algorithm, context)
    new_likelihood = Transformed(orig_likelihood, trafo, TDNoCorr())
    (result = PosteriorMeasure(new_likelihood, new_prior), trafo = trafo)
end


function bat_transform_impl(target::Union{PriorToUniform,PriorToGaussian}, density::Renormalized, algorithm::PriorSubstitution, context::BATContext)
    new_parent_density, trafo = bat_transform_impl(target, parent(density), algorithm, context)
    (result = Renormalized(new_parent_density, density.logrenormf), trafo = trafo)
end


# ToDo: Support bat_transform for vectors of variates and DensitySampleVector?


unshaping_trafo(::ArrayShape{Real, 1}) = identity
unshaping_trafo(vs::AbstractValueShape) = inverse(vs)


function transform_and_unshape(trafotarget::AbstractTransformTarget, object::Any, context::BATContext)
    orig_density = convert(AbstractMeasureOrDensity, object)
    trafoalg = bat_default(bat_transform, Val(:algorithm), trafotarget, orig_density)
    transformed_density, initial_trafo = bat_transform(trafotarget, orig_density, trafoalg, context)
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

Transform the variates of `smpls` using `f`.

As a convenience,

```julia
flat_smpls, f_flatten = bat_transform(Vector, smpls)
```

can be used to flatten the variates of `smpls` into
flat real-valued vectors.
"""
function bat_transform(f, smpls::DensitySampleVector, algorithm::TransformAlgorithm, context::BATContext)
    orig_context = deepcopy(context)
    r = bat_transform_impl(f, smpls, algorithm, context)
    result_with_args(r, (algorithm = algorithm, context = orig_context))
end


struct SampleTransformation <: TransformAlgorithm end

function bat_transform_impl(f::Function, smpls::DensitySampleVector, ::SampleTransformation, context::BATContext)
    (result = broadcast_arbitrary_trafo(f, smpls), trafo = f)
end

function bat_transform_impl(::Type{Vector}, smpls::DensitySampleVector, ::SampleTransformation, context::BATContext)
    shp = elshape(smpls.v)
    (result = unshaped.(smpls), trafo = inverse(shp))
end

function bat_transform_impl(shp::AbstractValueShape, smpls::DensitySampleVector, ::SampleTransformation, context::BATContext)
    (result = shp.(smpls), trafo = shp)
end
