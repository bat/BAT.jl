# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module BATPigeonsExt

using ArraysOfArrays: VectorOfSimilarVectors
using HeterogeneousComputing: get_rng
using InverseFunctions: inverse
using MeasureBase: massof
using Random: AbstractRNG, rand

import BAT
import Pigeons

BAT.pkgext(::Val{:Pigeons}) = BAT.PackageExtension{:Pigeons}()

struct BATPigeonsTarget{M,P}
    measure::M
    prior::P
end

(target::BATPigeonsTarget)(x) = BAT.checked_logdensityof(target.measure, x)

Pigeons.initialization(target::BATPigeonsTarget, rng::AbstractRNG, ::Int) =
    rand(rng, target.prior)

struct BATPigeonsReference{M}
    measure::M
end

(reference::BATPigeonsReference)(x) = BAT.checked_logdensityof(reference.measure, x)

function Pigeons.sample_iid!(reference::BATPigeonsReference, replica, shared)
    replica.state .= rand(replica.rng, reference.measure)
    return nothing
end

Pigeons.LogDensityProblems.logdensity(target::BATPigeonsTarget, x) = target(x)
Pigeons.LogDensityProblems.dimension(target::BATPigeonsTarget) =
    Int(BAT.some_dof(target.measure))
Pigeons.LogDensityProblems.logdensity(reference::BATPigeonsReference, x) = reference(x)
Pigeons.LogDensityProblems.dimension(reference::BATPigeonsReference) =
    Int(BAT.some_dof(reference.measure))

function _density_samples(pt, n_dof::Int)
    trace = Pigeons.get_sample(pt)
    isempty(trace) && throw(ArgumentError("Pigeons returned no target samples"))

    first_sample = first(trace)
    values = Matrix{eltype(first_sample)}(undef, n_dof, length(trace))
    logd = similar(first_sample, length(trace))

    for i in eachindex(trace)
        sample = trace[i]
        copyto!(view(values, :, i), 1, sample, 1, n_dof)
        logd[i] = sample[end]
    end

    return BAT.DensitySampleVector(v = VectorOfSimilarVectors(values), logd = logd)
end

function BAT.evalmeasure_impl(
    em::BAT.EvaluatedMeasure,
    algorithm::BAT.PigeonsSampling,
    context::BAT.BATContext,
)
    measure = BAT.unevaluated(em)
    measure isa BAT.AbstractPosteriorMeasure ||
        throw(ArgumentError("PigeonsSampling requires a posterior measure"))

    transformed_m, f_pretransform =
        BAT.transform_and_unshape(algorithm.pretransform, em, context)
    target = BAT.unevaluated(transformed_m)
    target isa BAT.AbstractPosteriorMeasure ||
        throw(ArgumentError("pretransform must preserve the posterior structure"))
    prior = BAT.getprior(target)
    n_dof = Int(BAT.some_dof(target))

    pt = Pigeons.pigeons(
        target = BATPigeonsTarget(target, prior),
        reference = BATPigeonsReference(prior),
        seed = rand(get_rng(context), 0:typemax(Int)),
        n_rounds = algorithm.n_rounds,
        n_chains = algorithm.n_chains,
        explorer = algorithm.explorer,
        multithreaded = algorithm.multithreaded,
        show_report = algorithm.show_report,
        record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
    )

    transformed_smpls = _density_samples(pt, n_dof)
    smpls = BAT.transform_samples(inverse(f_pretransform), transformed_smpls)
    ess = minimum(
        BAT.bat_eff_sample_size_impl(
            transformed_smpls.v,
            BAT.EffSampleSizeFromAC(),
            context,
        ).result,
    )
    lognormalizer_pair = Pigeons.stepping_stone_pair(pt)
    lognormalizer = Pigeons.stepping_stone(pt)
    mass = BAT._prior_importance_mass(exp(BAT.ULogarithmic, lognormalizer), massof(prior))
    diagnostics = (
        lognormalizer,
        lognormalizer_pair,
        n_tempered_restarts = Pigeons.n_tempered_restarts(pt),
        n_round_trips = Pigeons.n_round_trips(pt),
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
        mass,
        transformed = BAT._viewrep_measure(transformed_m, algorithm.pretransform),
        samplegen = nothing,
        evalinfo = BAT.MeasureEvalInfo(algorithm, diagnostics),
    )
end

end # module BATPigeonsExt
