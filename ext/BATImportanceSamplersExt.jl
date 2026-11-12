# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATImportanceSamplersExt

import AutoDiffOperators
import BAT
import ImportanceSamplers
using HeterogeneousComputing: CPUnit, get_compute_unit, get_rng
using InverseFunctions: inverse

BAT.pkgext(::Val{:ImportanceSamplers}) = BAT.PackageExtension{:ImportanceSamplers}()

function BAT.evalmeasure_impl(
    em::BAT.EvaluatedMeasure,
    algorithm::BAT.ImportanceSamplersSampling,
    context::BAT.BATContext,
)
    get_compute_unit(context) isa CPUnit || throw(
        ArgumentError("ImportanceSamplersSampling currently requires a CPU BATContext"),
    )
    transformed_m, f_pretransform =
        BAT.transform_and_unshape(algorithm.pretransform, em, context)
    target = BAT.unevaluated(transformed_m)
    logdensity = BAT.checked_logdensityof(target)
    ad = AutoDiffOperators.reverse_adtype(BAT.get_adselector(context))
    native = ImportanceSamplers.importance_sample(
        get_rng(context),
        ImportanceSamplers.LogTarget(logdensity, ad),
        algorithm.sampler;
        threaded = algorithm.threaded,
    )

    values = BAT.VectorOfSimilarVectors(native.samples)
    weights = ImportanceSamplers.normalized_weights(native)
    info =
        isempty(native.provenance) ? fill(nothing, length(native)) :
        BAT.StructArray(native.provenance)
    executor = algorithm.threaded ? BAT.MultiThreadedExec() : BAT.SequentialExec()
    transformed_smpls = BAT.DensitySampleVector(
        v = values,
        logd = BAT.exec_map(logdensity, executor, values),
        weight = weights,
        info = info,
    )
    smpls = BAT.transform_samples(inverse(f_pretransform), transformed_smpls, executor)
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
