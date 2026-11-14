# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATImportanceSamplersExt

import AutoDiffOperators
import Adapt
import BAT
import ImportanceSamplers
using ImportanceSamplers.KernelAbstractions: @kernel, @index, get_backend
using HeterogeneousComputing: CPUnit, get_compute_unit, get_rng, on_device
using InverseFunctions: inverse

BAT.pkgext(::Val{:ImportanceSamplers}) = BAT.PackageExtension{:ImportanceSamplers}()

struct BatchedBATTarget{T} <: Function
    target::T
end
Adapt.@adapt_structure BatchedBATTarget
(target::BatchedBATTarget)(x) = BAT.logdensityof(target.target, x)
function (target::BatchedBATTarget)(out, samples)
    out .= BAT.checked_logdensities(target.target, BAT.VectorOfSimilarVectors(samples))
    return nothing
end

# Keep Gaussian linear algebra outside scalar device likelihood calls.
const VectorGaussianMeasure =
    Union{BAT.AsMeasure{<:BAT.MvNormal},BAT.MeasureBase.StdPowerMeasure{BAT.StdNormal,1}}
const GaussianPosterior = BAT.PosteriorMeasure{L,<:VectorGaussianMeasure} where {L}
const GaussianLikelihood = BAT.FiniteVariateDensity{
    D,
    <:BAT.TransportFunction{<:VectorGaussianMeasure,<:VectorGaussianMeasure},
} where {D}

_score_likelihood(likelihood, samples) = likelihood, samples
function _score_likelihood(likelihood::GaussianLikelihood, samples)
    mapped = BAT.MeasureBase._gaussian_transport_matrix(likelihood.f_to_variate, samples)
    return BAT.FiniteVariateDensity(likelihood.likelihood, identity), mapped
end
_score_pullback(likelihood, gradient) = gradient
_score_pullback(likelihood::GaussianLikelihood, gradient) =
    BAT.MeasureBase._gaussian_transport_pullback(likelihood.f_to_variate, gradient)

function (f::BatchedBATTarget{<:GaussianPosterior})(out, samples)
    prior = BAT.MeasureBase.logdensities(BAT.getprior(f.target), samples)
    likelihood, mapped = _score_likelihood(BAT.getlikelihood(f.target), samples)
    out .= (
        (value, i) -> BAT._posterior_logval(
            likelihood,
            value,
            ImportanceSamplers._GradientSample(view(mapped, :, i)),
        )
    ).(
        prior,
        eachindex(prior),
    )
    return nothing
end

struct GaussianBatchGradient{T} <: ImportanceSamplers._BoundBatchGradient
    target::T
end
function ImportanceSamplers._prepare_accelerator_gradient(
    target::ImportanceSamplers._PreparedLogTarget{F,P,A,Nothing},
    locations,
    values,
) where {
    F<:BatchedBATTarget{<:GaussianPosterior},
    P,
    A<:ImportanceSamplers.ADTypes.AutoEnzyme,
}
    target = ImportanceSamplers._scalar_target(target)
    sample = ImportanceSamplers._GradientSample(view(locations, :, 1))
    ImportanceSamplers.ADTypes.mode(target.adtype) isa Union{
        ImportanceSamplers.ADTypes.ReverseMode,
        ImportanceSamplers.ADTypes.ForwardOrReverseMode,
    } || ImportanceSamplers._reject_cpu_gradient_on_accelerator(sample)
    ImportanceSamplers._bind_resolved_target(target, sample)
    return GaussianBatchGradient(target)
end

@kernel function gaussian_likelihood_score!(
    values,
    gradient,
    locations,
    prior,
    likelihood_ref,
    mode,
    workers,
)
    worker = @index(Global, Linear)
    for column = worker:workers:length(values)
        for i in axes(locations, 1)
            gradient[i, column] = zero(eltype(gradient))
        end
        sample = ImportanceSamplers._GradientSample(view(locations, :, column))
        shadow = ImportanceSamplers._GradientSample(view(gradient, :, column))
        likelihood, value = likelihood_ref[], prior[column]
        density = x -> BAT._posterior_logval(likelihood, value, x)
        values[column] = BAT.MeasureBase._device_logdensity_value_gradient!(
            mode,
            density,
            sample,
            shadow,
        )
    end
end

function ImportanceSamplers._batch_value_and_gradient!(
    values,
    gradient,
    bound::GaussianBatchGradient,
    locations,
)
    target = bound.target
    posterior = target.logdensity.target
    prior, score =
        BAT.MeasureBase._gaussian_logdensity_score(BAT.getprior(posterior), locations)
    original = BAT.getlikelihood(posterior)
    likelihood, mapped = _score_likelihood(original, locations)
    physical_gradient = similar(mapped)
    # Bound live AD tapes, not sample count. Workers traverse all columns on-device.
    workers = min(length(values), 8192)
    gaussian_likelihood_score!(get_backend(mapped), 64)(
        values,
        physical_gradient,
        mapped,
        prior,
        Ref(likelihood),
        target.adtype.mode,
        workers;
        ndrange = workers,
    )
    pulled = _score_pullback(original, physical_gradient)
    gradient .=
        ifelse.(reshape(isfinite.(values), 1, :), score .+ pulled, zero(eltype(gradient)))
    return nothing
end

struct RetainedBATTarget{T,F} <: Function
    target::T
    inverse::F
end
Adapt.@adapt_structure RetainedBATTarget

function (retained::RetainedBATTarget)(z)
    x, ladj = BAT.with_logabsdet_jacobian(retained.inverse, z)
    likelihood = BAT.getlikelihood(retained.target)
    cached = BAT.FiniteVariateDensity(likelihood.likelihood, Returns(x))
    value = BAT.checked_logdensityof(
        BAT.PosteriorMeasure(cached, BAT.getprior(retained.target)),
        z,
    )
    return value, x, ladj
end

function (retained::RetainedBATTarget)(out, coordinates, logjacobians, samples)
    mapped, ladjs =
        BAT.transports_with_ladj(retained.inverse, BAT.VectorOfSimilarVectors(samples))
    copy_coordinates!(coordinates, mapped)
    likelihood = BAT.getlikelihood(retained.target)
    cached = BAT.FiniteVariateDensity(likelihood.likelihood, identity)
    prior = BAT.MeasureBase.logdensities(BAT.getprior(retained.target), samples)
    out .= BAT._PosteriorLogvalAt(cached, mapped).(prior, LinearIndices(prior))
    logjacobians .= ladjs
    return nothing
end

copy_coordinates!(out::AbstractVector, values::AbstractVector) = copyto!(out, values)
copy_coordinates!(out::AbstractMatrix, values) = copyto!(out, BAT.flatview(values))
copy_coordinates!(out::NamedTuple, values::BAT.StructArray) =
    foreach(copy_coordinates!, out, BAT.StructArrays.components(values))

retain_coordinates(target, f; batched = false) = nothing
# CPU Gaussian maps already use one batched determinant calculation.
# GPU retention also batches the map needed by the likelihood.
function retain_coordinates(
    target::BAT.PosteriorMeasure,
    f::BAT.TransportFunction{
        <:BAT._GaussianTransportMeasure,
        <:BAT._GaussianTransportMeasure,
    },
    ;
    batched = false,
)
    return batched ? posterior_retention(target, f; batched) : nothing
end
retain_coordinates(
    target::BAT.PosteriorMeasure,
    f::BAT.TransportFunction;
    batched = false,
) = posterior_retention(target, f; batched)
function posterior_retention(target, f; batched)
    likelihood = BAT.getlikelihood(target)
    likelihood isa BAT.FiniteVariateDensity || return nothing
    likelihood.f_to_variate == f || return nothing
    sample = f(BAT.testvalue(BAT.MeasureBase.transport_source(f)))
    retained_shape_supported(sample) || return nothing
    evaluator = RetainedBATTarget(target, f)
    return ImportanceSamplers.RetainCoordinates(
        evaluator,
        sample;
        batch = batched ? evaluator : nothing,
    )
end

retained_shape_supported(sample) = false
retained_shape_supported(::Union{Number,AbstractVector{<:Number}}) = true
retained_shape_supported(sample::NamedTuple) =
    !isempty(sample) && all(retained_shape_supported, values(sample))

function restored_samples(native, transformed, f, executor)
    isnothing(native.retained) && return BAT.transform_samples(f, transformed, executor)
    retained = native.retained
    return BAT.DensitySampleVector(
        v = coordinate_batch(retained.coordinates),
        logd = native.logtargets .- retained.logjacobians,
        weight = deepcopy(transformed.weight),
        info = deepcopy(transformed.info),
        aux = deepcopy(transformed.aux),
    )
end

coordinate_batch(coordinates::AbstractVector) = coordinates
coordinate_batch(coordinates::AbstractMatrix) = BAT.VectorOfSimilarVectors(coordinates)
coordinate_batch(coordinates::NamedTuple) =
    BAT.StructArray(map(coordinate_batch, coordinates))

function BAT.evalmeasure_impl(
    em::BAT.EvaluatedMeasure,
    algorithm::BAT.ImportanceSamplersSampling,
    context::BAT.BATContext,
)
    cunit = get_compute_unit(context)
    cunit isa CPUnit && return sample_on_unit(em, algorithm, context)
    return on_device(() -> sample_on_unit(em, algorithm, context), cunit)()
end

function sample_on_unit(em, algorithm, context)
    cunit = get_compute_unit(context)
    transformed_m, f_pretransform =
        BAT.transform_and_unshape(algorithm.pretransform, em, context)
    target = BAT.unevaluated(transformed_m)
    logdensity = BAT.checked_logdensityof(target)
    ad = AutoDiffOperators.reverse_adtype(BAT.get_adselector(context))
    retain =
        retain_coordinates(target, inverse(f_pretransform); batched = !(cunit isa CPUnit))
    configured = if cunit isa CPUnit
        ImportanceSamplers.LogTarget(logdensity, ad; retain)
    else
        batched = BatchedBATTarget(target)
        ImportanceSamplers.LogTarget(batched, ad; batch = batched, retain)
    end
    prepared = ImportanceSamplers.prepare_sampler(
        get_rng(context),
        configured,
        algorithm.sampler;
        threaded = algorithm.threaded,
        save_logtargets = true,
    )
    native = ImportanceSamplers.importance_sample!(
        cunit isa CPUnit ? prepared : cunit.device(prepared),
    )

    values = BAT.VectorOfSimilarVectors(native.samples)
    weights = ImportanceSamplers.normalized_weights(native)
    info =
        isempty(native.provenance) ? fill(nothing, length(native)) :
        BAT.StructArray(native.provenance)
    executor =
        cunit isa CPUnit && algorithm.threaded ? BAT.MultiThreadedExec() :
        BAT.SequentialExec()
    transformed_smpls = BAT.DensitySampleVector(
        v = values,
        logd = native.logtargets,
        weight = weights,
        info = info,
    )
    inverse_transform = inverse(f_pretransform)
    if !(cunit isa CPUnit) && isnothing(native.retained)
        inverse_transform = cunit.device(inverse_transform)
    end
    smpls = restored_samples(native, transformed_smpls, inverse_transform, executor)
    n_dof = Int(BAT.some_dof(target))
    ess = BAT.bat_eff_sample_size_impl(smpls, BAT.KishESS(), context).result
    dsm = BAT.DensitySampleMeasure(smpls; dof = n_dof, ess)

    return BAT.EvaluatedMeasure(
        em;
        transform_intent = algorithm.pretransform,
        f_transform = BAT._viewrep_f(f_pretransform, algorithm.pretransform),
        empirical = BAT._viewrep_empirical(
            dsm,
            transformed_smpls,
            f_pretransform,
            algorithm.pretransform,
            n_dof,
            ess,
        ),
        dof = n_dof,
        mass = exp(BAT.ULogarithmic, ImportanceSamplers.lognormalizer(native)),
        transformed = BAT._viewrep_measure(transformed_m, algorithm.pretransform),
        samplegen = nothing,
        evalinfo = BAT.MeasureEvalInfo(algorithm, (; samples = native)),
    )
end

end # module BATImportanceSamplersExt
