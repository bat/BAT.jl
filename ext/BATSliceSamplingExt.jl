# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATSliceSamplingExt

using HeterogeneousComputing: get_rng
using InverseFunctions: inverse

import BAT
import SliceSampling

BAT.pkgext(::Val{:SliceSampling}) = BAT.PackageExtension{:SliceSampling}()
BAT.ext_default(::BAT.PackageExtension{:SliceSampling}, ::Val{:SAMPLER}) =
    SliceSampling.RandPermGibbs(SliceSampling.SliceSteppingOut(1.0))

struct BATSliceTarget{M}
    measure::M
end

SliceSampling.LogDensityProblems.logdensity(target::BATSliceTarget, x) =
    BAT.checked_logdensityof(target.measure, x)
SliceSampling.LogDensityProblems.dimension(target::BATSliceTarget) =
    Int(BAT.some_dof(target.measure))
SliceSampling.LogDensityProblems.capabilities(::Type{<:BATSliceTarget}) =
    SliceSampling.LogDensityProblems.LogDensityOrder{0}()

function BAT.evalmeasure_impl(
    em::BAT.EvaluatedMeasure,
    algorithm::BAT.SliceMCMCSampling,
    context::BAT.BATContext,
)
    transformed_m, f_pretransform =
        BAT.transform_and_unshape(algorithm.pretransform, em, context)
    target = BAT.unevaluated(transformed_m)
    n_dof = Int(BAT.some_dof(target))
    initalg = BAT.apply_trafo_to_init(f_pretransform, algorithm.init)
    init_params = collect(BAT.bat_initval(target, initalg, context).result)

    chain = SliceSampling.sample(
        get_rng(context),
        BATSliceTarget(target),
        algorithm.sampler,
        algorithm.nsamples;
        initial_params = init_params,
        discard_initial = algorithm.n_burnin,
        progress = false,
    )
    transformed_smpls = BAT.DensitySampleVector(
        v = getproperty.(chain, :params),
        logd = getproperty.(chain, :lp),
        info = getproperty.(chain, :info),
    )
    smpls = BAT.transform_samples(inverse(f_pretransform), transformed_smpls)
    ess = minimum(
        BAT.bat_eff_sample_size_impl(
            transformed_smpls.v,
            BAT.EffSampleSizeFromAC(),
            context,
        ).result,
    )
    dsm = BAT.DensitySampleMeasure(smpls, dof = n_dof, ess = ess)

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
        transformed = BAT._viewrep_measure(transformed_m, algorithm.pretransform),
        samplegen = nothing,
        evalinfo = BAT.MeasureEvalInfo(algorithm, (;)),
    )
end

end # module BATSliceSamplingExt
