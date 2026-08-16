# This file is a part of BAT.jl, licensed under the MIT License (MIT).
using BAT
using Test

using LinearAlgebra, Random, Distributions, Random123, ValueShapes
import AdvancedHMC, ForwardDiff

mutable struct MomentumCountingRNG{R<:AbstractRNG} <: AbstractRNG
    rng::R
    n_momentum_draws::Int
end

Random.rand(rng::MomentumCountingRNG, args...) = rand(rng.rng, args...)
Random.rand(rng::MomentumCountingRNG, ::Type{T}) where {T} = rand(rng.rng, T)
Random.rand!(rng::MomentumCountingRNG, args...) = rand!(rng.rng, args...)
Random.randn(rng::MomentumCountingRNG, args...) = randn(rng.rng, args...)
BAT.rngpart_getseed(rng::MomentumCountingRNG) = BAT.rngpart_getseed(rng.rng)
BAT.rngpart_getpartctrs(rng::MomentumCountingRNG) = BAT.rngpart_getpartctrs(rng.rng)
function BAT.rngpart_setpartctrs!(rng::MomentumCountingRNG, partctrs, depth)
    BAT.rngpart_setpartctrs!(rng.rng, partctrs, depth)
    return rng
end

function AdvancedHMC.rand_momentum(
    rng::MomentumCountingRNG,
    metric::AdvancedHMC.UnitEuclideanMetric,
    kinetic::AdvancedHMC.GaussianKinetic,
    position::AbstractVecOrMat,
)
    rng.n_momentum_draws += 1
    return AdvancedHMC.rand_momentum(rng.rng, metric, kinetic, position)
end

@testset "HMC walker momentum draws" begin
    nwalkers = 2
    rng = MomentumCountingRNG(Philox4x((0x548, 0)), 0)
    context = BATContext(rng = rng, ad = ForwardDiff)
    target = unshaped(batmeasure(MvNormal(zeros(2), I)))
    samplingalg = TransformedMCMC(
        proposal = HamiltonianMC(),
        transform_tuning = BAT.StanLikeTuning(),
        nwalkers = nwalkers,
    )
    initial_positions = [zeros(2) for _ in 1:nwalkers]
    mcmc_state = BAT.MCMCState(samplingalg, target, 1, initial_positions, context)
    chain_state = mcmc_state.chain_state
    n_momentum_draws_before = rng.n_momentum_draws

    BAT.mcmc_propose!!(chain_state, chain_state.proposal)

    @test rng.n_momentum_draws - n_momentum_draws_before == nwalkers
end
