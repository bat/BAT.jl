# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test
using BAT: NoAdaptiveTransform
using Distributions, LinearAlgebra, Random123, Statistics

const ENSEMBLE_TARGET_MEAN = [0.7, -1.2]
const ENSEMBLE_TARGET_COVARIANCE = [1.4 0.8; 0.8 2.0]

function ensemble_move_state(
    proposal, initial;
    target = batmeasure(MvNormal(ENSEMBLE_TARGET_MEAN, ENSEMBLE_TARGET_COVARIANCE)),
    seed,
)
    algorithm = TransformedMCMC(
        proposal = proposal,
        pretransform = DoNotTransform(),
        adaptive_transform = NoAdaptiveTransform(),
        convergence = AssumeConvergence(),
        nwalkers = length(initial),
        proposal_tuning = NoMCMCProposalTuning(),
        transform_tuning = NoMCMCTransformTuning(),
    )
    return BAT.MCMCState(
        algorithm,
        target,
        1,
        initial,
        BATContext(rng = Philox4x((573, seed))),
    )
end

function ensemble_move_samples(proposal, initial; seed, nwarmup, nsweeps)
    state = ensemble_move_state(proposal, initial; seed)
    state = BAT.mcmc_iterate!!(nothing, state; max_nsteps = nwarmup)
    outputs = BAT._empty_chain_outputs(state)
    state = BAT.mcmc_iterate!!(outputs, state; max_nsteps = nsweeps)
    return BAT._merge_chain_outputs(state, [outputs])
end

function elliptic_ensemble(
    nwalkers; mean = ENSEMBLE_TARGET_MEAN, covariance = ENSEMBLE_TARGET_COVARIANCE,
)
    factor = cholesky(Symmetric(covariance)).L
    return [mean + factor * [0.5cospi(2k / nwalkers), 0.5sinpi(2k / nwalkers)]
            for k in 0:(nwalkers - 1)]
end

function check_ensemble_gaussian_moments(proposal; seed)
    samples = ensemble_move_samples(
        proposal, elliptic_ensemble(8); seed, nwarmup = 400, nsweeps = 800,
    )
    @test maximum(abs, mean(samples) - ENSEMBLE_TARGET_MEAN) < 0.18
    @test maximum(abs, cov(samples) - ENSEMBLE_TARGET_COVARIANCE) < 0.35
end
