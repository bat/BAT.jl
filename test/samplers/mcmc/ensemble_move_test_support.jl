# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using BAT: NoAdaptiveTransform
using DensityInterface, Distributions, LinearAlgebra, Random123, Statistics, ValueShapes


struct _CountingEnsembleTarget{M,F,C} <: BAT.BATMeasure
    base::M
    logdensity::F
    calls::C
end

_increment_ensemble_calls!(calls::Base.RefValue{Int}) = (calls[] += 1)
_increment_ensemble_calls!(calls::Threads.Atomic{Int}) = Threads.atomic_add!(calls, 1)

DensityInterface.logdensityof(target::_CountingEnsembleTarget, x) =
    (_increment_ensemble_calls!(target.calls); target.logdensity(x))
ValueShapes.varshape(target::_CountingEnsembleTarget) = ValueShapes.varshape(target.base)


struct _FailingEnsembleTarget{M,F} <: BAT.BATMeasure
    base::M
    logdensity::F
    calls::Base.RefValue{Int}
    fail_at::Base.RefValue{Int}
end

function DensityInterface.logdensityof(target::_FailingEnsembleTarget, x)
    target.calls[] += 1
    target.calls[] == target.fail_at[] && error("controlled ensemble target failure")
    return target.logdensity(x)
end
ValueShapes.varshape(target::_FailingEnsembleTarget) = ValueShapes.varshape(target.base)


function _capture_ensemble_error(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end


function _ensemble_move_state(
    proposal,
    v_init;
    nwalkers = length(v_init),
    target = let
        T = float(eltype(first(v_init)))
        d = length(first(v_init))
        batmeasure(MvNormal(zeros(T, d), Diagonal(ones(T, d))))
    end,
    adaptive_transform = NoAdaptiveTransform(),
    sample_weighting = RepetitionWeighting(),
    proposal_tuning = NoMCMCProposalTuning(),
    transform_tuning = NoMCMCTransformTuning(),
    rng_seed = (564, 80),
)
    algorithm = TransformedMCMC(
        proposal = proposal,
        pretransform = DoNotTransform(),
        adaptive_transform = adaptive_transform,
        convergence = AssumeConvergence(),
        nwalkers = nwalkers,
        sample_weighting = sample_weighting,
        proposal_tuning = proposal_tuning,
        transform_tuning = transform_tuning,
    )
    return BAT.MCMCState(
        algorithm, target, 1, v_init, BATContext(rng = Philox4x(rng_seed)),
    )
end


function _run_ensemble_move(
    proposal,
    target,
    v_init;
    seed::Integer,
    nwarmup::Integer,
    nsweeps::Integer,
)
    state = _ensemble_move_state(
        proposal, v_init; target, rng_seed = (564, seed),
    )
    state = BAT.mcmc_iterate!!(nothing, state; max_nsteps = nwarmup)
    outputs = BAT._empty_chain_outputs(state)
    state = BAT.mcmc_iterate!!(outputs, state; max_nsteps = nsweeps)
    samples = BAT._merge_chain_outputs(state, [outputs])
    return (; state, outputs, samples)
end

function _two_dimensional_elliptic_initial_ensemble(
    mean,
    covariance,
    nwalkers::Integer;
    radius = 1.5,
)
    scale = cholesky(Symmetric(covariance)).L
    return [
        mean + scale * [
            radius * cospi(2 * k / nwalkers),
            radius * sinpi(2 * k / nwalkers),
        ]
        for k in 0:(nwalkers - 1)
    ]
end

function _check_ensemble_gaussian_moments(
    proposal;
    seed,
    mean_tolerance,
    covariance_tolerance,
)
    mean_target = [0.7, -1.2]
    covariance_target = [1.4 0.8; 0.8 2.0]
    nwalkers = 16
    initial = _two_dimensional_elliptic_initial_ensemble(
        mean_target, covariance_target, nwalkers; radius = 0.5,
    )
    result = _run_ensemble_move(
        proposal, batmeasure(MvNormal(mean_target, covariance_target)), initial;
        seed, nwarmup = 500, nsweeps = 1500,
    )

    @test sum(result.samples.weight) == nwalkers * 1500
    @test maximum(abs, mean(result.samples) - mean_target) < mean_tolerance
    @test maximum(abs, cov(result.samples) - covariance_target) <
        covariance_tolerance
end
