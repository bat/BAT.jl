# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, Statistics
using Distributions, ValueShapes
using StableRNGs
import ForwardDiff
import FunctionChains
using AffineMaps: MulAdd

using BAT: batmeasure, TriangularAffineTransform, NoMCMCTransformTuning

@testset "multitrafo_tuner" begin
    rng = StableRNG(559102837)
    context = BATContext(ad = ForwardDiff)

    objective = MvNormal([1.0, -1.0], [2.0 1.2; 1.2 1.5])
    target = batmeasure(objective)

    # An untuned outer component (prior-based init) plus a RAM-tuned
    # inner component starting from the identity: the composite geometry
    # has to be learned by the inner component alone:
    at = AdaptiveTransformChain((
        TriangularAffineTransform(init = BAT.UnitTransformInit()),
        TriangularAffineTransform(),
    ))
    alg = TransformedMCMC(
        proposal = RandomWalk(),
        adaptive_transform = at,
        transform_tuning = MultiTrafoTuning((RAMTuning(), NoMCMCTransformTuning())),
        pretransform = DoNotTransform(),
        nchains = 2,
        nsteps = 5 * 10^4
    )

    smplres = BAT.sample_and_verify(target, alg, objective, context)
    @test smplres.verified

    # The tuned transform is a rebuilt chain (component updates are
    # functional, so the tuning orchestration resynchronized the chain
    # state on every change) whose composite geometry matches the target:
    cs = BAT.samplegenof(smplres.evaluated).chain_states[1]
    comps = FunctionChains.fchainfs(cs.f_transform)
    @test length(comps) == 2
    @test all(c -> c isa MulAdd, comps)
    A_eff = comps[2].A * comps[1].A
    Σ = cov(objective)
    G_learned = Matrix(A_eff * A_eff')
    @test opnorm(G_learned - Σ) / opnorm(Σ) < 0.5

    # Score-based tunings are not supported inside transform chains yet:
    # the default configuration for gradient-based proposals must already
    # fail at construction time (not later during state creation) ...
    @test_throws ArgumentError TransformedMCMC(
        proposal = HamiltonianMC(),
        adaptive_transform = at,
        pretransform = DoNotTransform()
    )
    # ... and explicitly configured Fisher components are rejected at
    # tuner creation:
    alg_hmc = TransformedMCMC(
        proposal = HamiltonianMC(),
        adaptive_transform = at,
        transform_tuning = MultiTrafoTuning((FisherTransformTuning(), FisherTransformTuning())),
        pretransform = DoNotTransform()
    )
    @test_throws ArgumentError BAT.MCMCState(alg_hmc, target, 1, [randn(rng, 2)], deepcopy(context))

    # Component count and tuning count must match:
    alg_mismatch = TransformedMCMC(
        proposal = RandomWalk(),
        adaptive_transform = at,
        transform_tuning = MultiTrafoTuning((RAMTuning(),)),
        pretransform = DoNotTransform()
    )
    @test_throws ArgumentError BAT.MCMCState(alg_mismatch, target, 1, [randn(rng, 2)], deepcopy(context))
end
