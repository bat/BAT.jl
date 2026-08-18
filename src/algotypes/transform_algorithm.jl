# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type TransformAlgorithm

Abstract type for density transformation algorithms.
"""
abstract type TransformAlgorithm end
export TransformAlgorithm


"""
    bat_transform(
        how::TransformIntent,
        object,
        [algorithm::TransformAlgorithm]
    )

    bat_transform(
        f,
        object,
        [algorithm::TransformAlgorithm]
    )

Transform `object` to another variate space: either as implied by the
[`TransformIntent`](@ref) `how` together with `object`, or using a given
invertible transformation function `f` directly.

Returns a NamedTuple of the shape

```julia
(result = newdensity, f_transform = vartrafo::Function, ...)
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
_convert_trafo_how(::Type{<:Vector}) = TransformIntent(Vector)

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
    struct IdentityTransformAlgorithm <: TransformAlgorithm

A no-op density transform algorithm that leaves any density unchanged.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct IdentityTransformAlgorithm <: TransformAlgorithm end
export IdentityTransformAlgorithm


function bat_transform_impl(::DoNotTransform, measure::MeasureLike, ::IdentityTransformAlgorithm, ::BATContext)
    (result = batmeasure(measure), f_transform = identity)
end


_distmeasure_trafo(intent::UniformBased, density::BATDistMeasure) = DistributionTransform(Uniform, Distribution(density))

function bat_transform_impl(intent::UniformBased, density::BATDistMeasure{<:StandardUniformDist}, algorithm::IdentityTransformAlgorithm, context::BATContext)
    (result = density, f_transform = identity)
end


_distmeasure_trafo(intent::NormalBased, density::BATDistMeasure) = DistributionTransform(Normal, Distribution(density))

function bat_transform_impl(intent::NormalBased, density::BATDistMeasure{<:StandardNormalDist}, algorithm::IdentityTransformAlgorithm, context::BATContext)
    (result = density, f_transform = identity)
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
_get_deep_prior_for_trafo(em::EvaluatedMeasure) = _get_deep_prior_for_trafo(unevaluated(em))
# The implied transformation is invariant under reweighting:
_get_deep_prior_for_trafo(m::BATWeightedMeasure) = _get_deep_prior_for_trafo(m.base)


"""
    BAT.transform_function(intent::TransformIntent, object)

*BAT-internal, not part of stable public API.*

Return the transformation function that `intent` implies for `object`.

A [`TransformIntent`](@ref), together with an object to be transformed,
implies a concrete transformation function; the same intent and object
always yield the same transformation. Methods of `transform_function` must
derive the transformation from the intent and the object alone.
"""
function transform_function end

transform_function(::DoNotTransform, ::Any) = identity

transform_function(::ToRealVector, obj::Union{BATMeasure,DensitySampleVector}) = Base.Fix2(unshaped, varshape(obj))

function transform_function(intent::Union{UniformBased,NormalBased}, m::BATMeasure)
    _distmeasure_trafo(intent, _get_deep_prior_for_trafo(m))
end

function transform_function(intent::Union{UniformBased,NormalBased}, m::BATPushFwdMeasure)
    ffcomp(transform_function(intent, m.origin), m.finv)
end

transform_function(intent::Union{UniformBased,NormalBased}, m::BATWeightedMeasure) = transform_function(intent, m.base)


function bat_transform_impl(intent::Union{UniformBased,NormalBased}, m::AbstractPosteriorMeasure, algorithm::FullMeasureTransform, context::BATContext)
    f_transform = transform_function(intent, m)
    (result = BATPushFwdMeasure(f_transform, m, KeepRootMeasure()), f_transform = f_transform)
end


function bat_transform_impl(intent::Union{UniformBased,NormalBased}, m::BATDistMeasure, algorithm::FullMeasureTransform, context::BATContext)
    f_transform = transform_function(intent, m)
    (result = BATPushFwdMeasure(f_transform, m, KeepRootMeasure()), f_transform = f_transform)
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


function bat_transform_impl(intent::Union{UniformBased,NormalBased}, density::BATDistMeasure, algorithm::PriorSubstitution, context::BATContext)
    f_transform = transform_function(intent, density)
    transformed_density = BATDistMeasure(f_transform.target_dist)
    (result = transformed_density, f_transform = f_transform)
end


function bat_transform_impl(intent::TransformIntent, m::BATPushFwdMeasure, algorithm::PriorSubstitution, context::BATContext)
    new_measure, f_transform_orig = bat_transform_impl(intent, m.origin, algorithm, context)
    f_transform = ffcomp(f_transform_orig, m.finv)
    (result = new_measure, f_transform = f_transform)
end


# The implied transformation is invariant under reweighting, and the weight
# must be preserved by prior substitution:
function bat_transform_impl(intent::Union{UniformBased,NormalBased}, m::BATWeightedMeasure, algorithm::PriorSubstitution, context::BATContext)
    tr = bat_transform_impl(intent, m.base, algorithm, context)
    (result = weightedmeasure(m.logweight, tr.result), f_transform = tr.f_transform)
end


function bat_transform_impl(intent::Union{UniformBased,NormalBased}, density::AbstractPosteriorMeasure, algorithm::PriorSubstitution, context::BATContext)
    orig_prior = getprior(density)
    orig_likelihood = getlikelihood(density)
    new_prior, f_transform = bat_transform_impl(intent, orig_prior, algorithm, context)
    new_likelihood = _precompose_density(orig_likelihood, inverse(f_transform))
    (result = PosteriorMeasure(new_likelihood, new_prior), f_transform = f_transform)
end


function bat_transform_impl(intent::TransformIntent, em::EvaluatedMeasure, algorithm::PriorSubstitution, context::BATContext)
    new_measure, f_transform = bat_transform_impl(intent, unevaluated(em), algorithm, context)
    annexes_match = _intents_match(em.transform_intent, intent)
    em_f_hash = hash(em.f_transform)
    new_empirical = _transformed_empirical(annexes_match, em_f_hash, _empirical_rep(em), f_transform, context)
    # Modes refer to the untransformed space (the log-abs-det-Jacobian shifts
    # maximizers), so they can't be carried over. The approximation and the
    # sample generator carry over exactly when their space matches the intent:
    new_approx = _transformed_approx(annexes_match, em_f_hash, em.approx)
    new_samplegen = _transformed_samplegen(annexes_match, em.samplegen)
    new_em = EvaluatedMeasure(
        BispacedMeasure(new_measure), DoNotTransform(), identity, new_empirical, new_approx,
        em.dof, em.mass, nothing, new_samplegen, nothing
    )
    (result = new_em, f_transform = f_transform)
end

_transformed_empirical(::Bool, ::UInt, ::Nothing, f_transform, ::BATContext) = nothing

# A matching pre-transformed representation makes the sample transport free.
# Its transformation-hash witness is verified before it is served:
function _transformed_empirical(annexes_match::Bool, em_f_hash::UInt, p::BispacedMeasure, f_transform, context::BATContext)
    if annexes_match && !isnothing(p.transformed)
        p.f_hash == em_f_hash || _throw_pair_hash_mismatch("Empirical pair")
        BispacedMeasure(p.transformed)
    else
        smpl_trafoalg = bat_default(bat_transform, Val(:algorithm), f_transform, p.main)
        BispacedMeasure(bat_transform_impl(f_transform, p.main, smpl_trafoalg, context).result)
    end
end

# On a matching intent, the transformed-space side of the approximation pair
# is directly the approximation of the transformed measure. On a miss there
# is no safe way to recover it (stripping a pushforward would require
# comparing transform functions by value), so it is dropped:
_transformed_approx(::Bool, ::UInt, ::Nothing) = nothing

function _transformed_approx(annexes_match::Bool, em_f_hash::UInt, p::BispacedMeasure)
    if annexes_match && !isnothing(p.transformed)
        p.f_hash == em_f_hash || _throw_pair_hash_mismatch("Approximation pair")
        BispacedMeasure(p.transformed)
    else
        nothing
    end
end

# A sample generator is process state native to the transformed-space view
# it was produced in: it carries over on a matching intent and is dropped
# otherwise, since no transported representation of it exists:
_transformed_samplegen(::Bool, ::Nothing) = nothing
_transformed_samplegen(annexes_match::Bool, gen::AbstractSampleGenerator) = annexes_match ? gen : nothing


# ToDo: Support bat_transform for vectors of variates and DensitySampleVector?


# ToDo: Remove transform_and_unshape and use `ToRealVector` instead of `DoNotTransform` in algorithms?
function transform_and_unshape(intent::TransformIntent, object::Any, context::BATContext)
    orig_measure = batmeasure(object)
    fast_result = _transform_and_unshape_cached(orig_measure, intent)
    isnothing(fast_result) || return fast_result
    trafoalg = bat_default(bat_transform, Val(:algorithm), intent, orig_measure)
    tr1 = bat_transform(intent, orig_measure, trafoalg, context)
    tr2 = bat_transform(ToRealVector(), tr1.result, UnshapeTransformation(), context)
    result_trafo = ffcomp(tr2.f_transform, tr1.f_transform)
    return _keep_transformed_identity(orig_measure, tr2.result, intent),
        _keep_f_identity(orig_measure, result_trafo, intent)
end

# With a matching view and complete caches the transformed representation
# can be assembled directly, without re-deriving the transformation:
_transform_and_unshape_cached(::BATMeasure, ::TransformIntent) = nothing
_transform_and_unshape_cached(::BATMeasure, ::DoNotTransform) = nothing
_transform_and_unshape_cached(::EvaluatedMeasure, ::DoNotTransform) = nothing

function _transform_and_unshape_cached(em::EvaluatedMeasure, intent::TransformIntent)
    em_f_hash = hash(em.f_transform)
    if _intents_match(em.transform_intent, intent) &&
            !isnothing(em.unevaluated.transformed) && !isnothing(em.f_transform) &&
            !_pair_claims_mismatch(em.unevaluated, em_f_hash) &&
            !_pair_claims_mismatch(em.approx, em_f_hash) &&
            (isnothing(em.empirical) || (_has_pair_annex(em.empirical) && !_pair_claims_mismatch(em.empirical, em_f_hash)))
        new_em = EvaluatedMeasure(
            BispacedMeasure(em.unevaluated.transformed), DoNotTransform(), identity,
            _flip_empirical_annex(em.empirical), _transformed_approx(true, em_f_hash, em.approx),
            em.dof, em.mass, nothing, _transformed_samplegen(true, em.samplegen), nothing
        )
        return (new_em, em.f_transform)
    else
        return nothing
    end
end

_flip_empirical_annex(::Nothing) = nothing
_flip_empirical_annex(p::BispacedMeasure) = BispacedMeasure(p.transformed)


# Producers that work in a transformed space report their view content via
# these helpers, stamped with the transformation used. With DoNotTransform
# both spaces coincide, so nothing extra is stored:
_viewrep_empirical(dsm::DensitySampleMeasure, ::DensitySampleVector, ::Any, ::DoNotTransform, n_dof, ess) = dsm

function _viewrep_empirical(dsm::DensitySampleMeasure, smpls_z::DensitySampleVector, f_pretransform::Any, ::TransformIntent, n_dof, ess)
    BispacedMeasure(dsm, DensitySampleMeasure(smpls_z, dof = n_dof, ess = ess), hash(f_pretransform))
end

_viewrep_measure(::BATMeasure, ::DoNotTransform) = unchanged
_viewrep_measure(transformed_m::BATMeasure, ::TransformIntent) = unevaluated(transformed_m)

_viewrep_f(::Any, ::DoNotTransform) = unchanged
_viewrep_f(f_pretransform::Any, ::TransformIntent) = f_pretransform

# A cached bare measure in the transformed space replaces the freshly
# constructed, structurally equal one, so that repeated evaluations with the
# same intent see the identical object (which keeps compiled artifacts like
# AD preparations valid):
_keep_transformed_identity(::BATMeasure, result_measure, ::TransformIntent) = result_measure

function _keep_transformed_identity(orig_em::EvaluatedMeasure, result_measure, intent::TransformIntent)
    p = orig_em.unevaluated
    cache_usable = _intents_match(orig_em.transform_intent, intent) &&
        !_pair_claims_mismatch(p, hash(orig_em.f_transform))
    cached = cache_usable ? p.transformed : nothing
    isnothing(cached) ? result_measure : _replace_unevaluated(result_measure, cached)
end

# DoNotTransform is the no-view sentinel, no cache exists for it (and none
# may be honored, even on a contract-violating EvaluatedMeasure):
_keep_transformed_identity(::EvaluatedMeasure, result_measure, ::DoNotTransform) = result_measure

_replace_unevaluated(::BATMeasure, cached::BATMeasure) = cached

function _replace_unevaluated(result_em::EvaluatedMeasure, cached::BATMeasure)
    p = result_em.unevaluated
    EvaluatedMeasure(
        BispacedMeasure(cached, p.transformed, p.f_hash), result_em.transform_intent, result_em.f_transform,
        result_em.empirical, result_em.approx, result_em.dof, result_em.mass,
        result_em.modes, result_em.samplegen, result_em.evalinfo
    )
end

# Likewise, a cached transformation function replaces the freshly resolved,
# value-equal one. DoNotTransform is the no-view sentinel (its cached
# function is `identity` by convention), there the freshly resolved
# unshaping function is the correct result:
_keep_f_identity(::BATMeasure, result_trafo, ::TransformIntent) = result_trafo

function _keep_f_identity(orig_em::EvaluatedMeasure, result_trafo, intent::TransformIntent)
    f = _intents_match(orig_em.transform_intent, intent) ? orig_em.f_transform : nothing
    isnothing(f) ? result_trafo : f
end

_keep_f_identity(::EvaluatedMeasure, result_trafo, ::DoNotTransform) = result_trafo



"""
    struct SampleTransformation <: TransformAlgorithm

*BAT-internal, not part of stable public API.*
"""
struct SampleTransformation <: TransformAlgorithm end

function bat_transform_impl(f, dsm::DensitySampleMeasure, algorithm::SampleTransformation, context::BATContext)
    smpls = samplesof(dsm)
    new_smpls = bat_transform_impl(f, smpls, algorithm, context).result
    new_dsm = DensitySampleMeasure(new_smpls, dof = dsm._dof, ess = dsm._ess, mass = massof(dsm))
    (result = new_dsm, f_transform = f)
end

function bat_transform_impl(f::Function, smpls::DensitySampleVector, ::SampleTransformation, context::BATContext)
    (result = transform_samples(f, smpls), f_transform = f)
end

function bat_transform_impl(shp::AbstractValueShape, smpls::DensitySampleVector, ::SampleTransformation, context::BATContext)
    (result = shp.(smpls), f_transform = shp)
end


"""
    struct UnshapeTransformation <: TransformAlgorithm

*BAT-internal, not part of stable public API.*
"""
struct UnshapeTransformation <: TransformAlgorithm end

function bat_transform_impl(::ToRealVector, obj::Union{BATMeasure,DensitySampleVector}, ::UnshapeTransformation, context::BATContext)
    f_transform = Base.Fix2(unshaped, varshape(obj))
    trafoalg = bat_default(bat_transform, Val(:algorithm), f_transform, obj)
    bat_transform_impl(f_transform, obj, trafoalg, context)
end

function bat_transform_impl(::Base.Fix2{typeof(unshaped),<:ArrayShape{<:Real,1}}, m::BATMeasure, ::FullMeasureTransform, context::BATContext)
    (result = m, f_transform = identity)
end

function bat_transform_impl(f::Base.Fix2{typeof(unshaped)}, m::BATMeasure, ::FullMeasureTransform, context::BATContext)
    shp = f.x
    (result = unshaped(m, shp), f_transform = f)
end

function bat_transform_impl(f::Base.Fix2{typeof(unshaped)}, smpls::DensitySampleVector, ::SampleTransformation, context::BATContext)
    unshape_vs = f.x
    @argcheck elshape(smpls.v) <= unshape_vs
    (result = unshaped.(smpls), f_transform = f)
end
