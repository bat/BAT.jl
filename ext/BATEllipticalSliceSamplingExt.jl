# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATEllipticalSliceSamplingExt

using Distributions: Distribution, logpdf
using HeterogeneousComputing: get_rng
using InverseFunctions: inverse

import BAT
import EllipticalSliceSampling

BAT.pkgext(::Val{:EllipticalSliceSampling}) =
    BAT.PackageExtension{:EllipticalSliceSampling}()

function BAT.evalmeasure_impl(
    em::BAT.EvaluatedMeasure,
    algorithm::BAT.EllipticalSliceMCMCSampling,
    context::BAT.BATContext,
)
    BAT.unevaluated(em) isa BAT.AbstractPosteriorMeasure ||
        throw(ArgumentError("EllipticalSliceMCMCSampling requires a posterior measure"))

    intent = BAT.NormalBased()
    transformed_m, f_pretransform = BAT.transform_and_unshape(intent, em, context)
    target = BAT.unevaluated(transformed_m)
    target isa BAT.AbstractPosteriorMeasure || throw(
        ArgumentError("NormalBased transformation must preserve the posterior structure"),
    )
    prior_measure = BAT.getprior(target)
    prior_measure isa BAT.BATDistMeasure || throw(
        ArgumentError(
            "The posterior prior must support transformation to a Gaussian probability distribution",
        ),
    )
    prior = Distribution(prior_measure)
    likelihood = BAT.getlikelihood(target)
    model = EllipticalSliceSampling.ESSModel(
        prior,
        x -> BAT.checked_logdensityof(likelihood, x),
    )
    initalg = BAT.apply_trafo_to_init(f_pretransform, algorithm.init)
    initial_params = collect(BAT.bat_initval(target, initalg, context).result)
    T = typeof(model.loglikelihood(initial_params) + logpdf(prior, initial_params))
    logd = T[]
    callback =
        (_rng, _model, _sampler, sample, state, _iteration; kwargs...) ->
            push!(logd, state.loglikelihood + logpdf(prior, sample))

    samples = EllipticalSliceSampling.sample(
        get_rng(context),
        model,
        EllipticalSliceSampling.ESS(),
        algorithm.nsamples;
        initial_params,
        discard_initial = algorithm.n_burnin,
        progress = false,
        callback,
    )
    transformed_smpls = BAT.DensitySampleVector(v = samples, logd = logd)
    smpls = BAT.transform_samples(inverse(f_pretransform), transformed_smpls)
    n_dof = Int(BAT.some_dof(target))
    ess = minimum(
        BAT.bat_eff_sample_size_impl(
            transformed_smpls.v,
            BAT.EffSampleSizeFromAC(),
            context,
        ).result,
    )
    dsm = BAT.DensitySampleMeasure(smpls, dof = n_dof, ess = ess)

    BAT.EvaluatedMeasure(
        em;
        transform_intent = intent,
        f_transform = BAT._viewrep_f(f_pretransform, intent),
        empirical = BAT._viewrep_empirical(
            dsm,
            transformed_smpls,
            f_pretransform,
            intent,
            n_dof,
            ess,
        ),
        dof = n_dof,
        transformed = BAT._viewrep_measure(transformed_m, intent),
        samplegen = nothing,
        evalinfo = BAT.MeasureEvalInfo(algorithm, (;)),
    )
end

end # module BATEllipticalSliceSamplingExt
