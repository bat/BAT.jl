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

struct BATPigeonsTarget{M,I}
    measure::M
    initvals::I
end

(target::BATPigeonsTarget)(x) = BAT.checked_logdensityof(target.measure, x)

Pigeons.initialization(target::BATPigeonsTarget{<:Any,Nothing}, rng::AbstractRNG, ::Int) =
    rand(rng, BAT.getprior(target.measure))

Pigeons.initialization(target::BATPigeonsTarget, ::AbstractRNG, replica_index::Int) =
    collect(target.initvals[replica_index])

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
    n_samples = length(trace)

    first_sample = first(trace)
    values = Matrix{eltype(first_sample)}(undef, n_dof, n_samples)
    logd = similar(first_sample, n_samples)

    for i in 1:n_samples
        sample = trace[i]
        copyto!(view(values, :, i), 1, sample, 1, n_dof)
        logd[i] = sample[end]
    end

    return BAT.DensitySampleVector(v = VectorOfSimilarVectors(values), logd = logd)
end

function _leg_diagnostics(pt, tempering, edges)
    return (
        adapted_betas = copy(tempering.schedule.grids),
        swap_acceptance_pr = [
            Pigeons.value(pt.reduced_recorders.swap_acceptance_pr[(i, i + 1)])
            for i in edges
        ],
        global_barrier = Pigeons.global_barrier(tempering),
    )
end

_tempering_diagnostics(pt, tempering::Pigeons.NonReversiblePT) =
    _leg_diagnostics(pt, tempering, 1:(length(tempering.schedule.grids) - 1))

function _tempering_diagnostics(pt, tempering::Pigeons.StabilizedPT)
    n_var = length(tempering.variational_leg.schedule.grids)
    n_total = n_var + length(tempering.fixed_leg.schedule.grids)
    return (;
        _leg_diagnostics(pt, tempering.fixed_leg, (n_total - 1):-1:(n_var + 1))...,
        variational = _leg_diagnostics(pt, tempering.variational_leg, 1:(n_var - 1)),
    )
end

function BAT.evalmeasure_impl(
    em::BAT.EvaluatedMeasure,
    algorithm::BAT.PigeonsSampling,
    context::BAT.BATContext,
)
    algorithm.n_chains >= 2 || throw(ArgumentError("PigeonsSampling requires n_chains >= 2"))
    algorithm.n_rounds >= 1 || throw(ArgumentError("PigeonsSampling requires n_rounds >= 1"))

    n_var = algorithm.n_chains_variational
    n_var == 0 || (n_var >= 2 && algorithm.variational !== nothing) ||
        throw(ArgumentError("n_chains_variational requires a variational reference and at least two temperatures"))

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

    # Preserve Pigeons' replica RNG streams for prior initialization.
    initvals = if algorithm.init isa BAT.InitFromTarget && isnothing(BAT.empiricalof(em))
        nothing
    else
        original = BAT.bat_initval(em, algorithm.n_chains + n_var, algorithm.init, context).result
        BAT.transform_samples(f_pretransform, original)
    end

    variational = deepcopy(algorithm.variational)
    reference_prior = if isnothing(variational)
        prior
    else
        prior_mass = massof(prior)
        prior_mass isa Real && isfinite(log(prior_mass)) ||
            throw(ArgumentError("Variational PigeonsSampling requires a finite positive prior mass"))
        BAT.weightedmeasure(-log(prior_mass), prior)
    end
    # Pigeons' round-trip recorder assumes equal stabilized leg lengths.
    record_round_trip = n_var == 0 || n_var == algorithm.n_chains
    record = [Pigeons.traces; Pigeons.record_default()]
    record_round_trip && insert!(record, 2, Pigeons.round_trip)

    pt = Pigeons.pigeons(;
        target = BATPigeonsTarget(target, initvals),
        reference = BATPigeonsReference(reference_prior),
        seed = rand(get_rng(context), 0:typemax(Int)),
        n_rounds = algorithm.n_rounds,
        n_chains = algorithm.n_chains,
        n_chains_variational = n_var,
        variational,
        explorer = algorithm.explorer,
        multithreaded = algorithm.multithreaded,
        show_report = algorithm.show_report,
        record,
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
    reference_mass = isnothing(variational) ? massof(prior) : 1
    mass = BAT._prior_importance_mass(exp(BAT.ULogarithmic, lognormalizer), reference_mass)
    diagnostics = (;
        lognormalizer,
        lognormalizer_pair,
        _tempering_diagnostics(pt, pt.shared.tempering)...,
        n_tempered_restarts = record_round_trip ? Pigeons.n_tempered_restarts(pt) : missing,
        n_round_trips = record_round_trip ? Pigeons.n_round_trips(pt) : missing,
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
