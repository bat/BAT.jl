# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, LinearAlgebra, Random, Random123
using HeterogeneousComputing: get_rng, has_rng_streamkey, rng_streamkey, rootrng
using BAT: NoAdaptiveTransform, NoMCMCProposalTuning, NoMCMCTransformTuning


@testset "rng_streams" begin
    @testset "BATContext RNG" begin
        @test has_rng_streamkey(get_rng(BATContext()))
        for rng in (rootrng(7), Xoshiro(3), Philox4x((564, 1)))
            @test get_rng(BATContext(rng = rng)) === rng
            @test get_rng(BAT.set_rng(BATContext(), rng)) === rng
        end

        Random.seed!(42)
        key1 = rng_streamkey(get_rng(BATContext()))
        Random.seed!(42)
        key2 = rng_streamkey(get_rng(BATContext()))
        @test key1 == key2
    end

    target = batmeasure(MvNormal(zeros(2), I))

    @testset "MCMC random streams" begin
        algorithm = TransformedMCMC(
            proposal = RandomWalk(),
            proposal_tuning = NoMCMCProposalTuning(),
            pretransform = DoNotTransform(),
            adaptive_transform = NoAdaptiveTransform(),
            transform_tuning = NoMCMCTransformTuning(),
            tempering = BAT.NoMCMCTempering(),
            nwalkers = 2,
            nonzero_weights = false,
        )
        positions = [[0.0, 0.0], [1.0, 1.0]]

        function chain_path(context, between_iterations = identity)
            state = BAT.MCMCState(algorithm, target, 1, deepcopy(positions), context)
            outputs = BAT._empty_chain_outputs(state)
            for _ in 1:3
                between_iterations(context)
                BAT.mcmc_iterate!!(outputs, state; max_nsteps = 2, nonzero_weights = false)
            end
            return [copy.(output.v) for output in outputs]
        end

        context = BATContext(rng = rootrng(5643))
        reference = chain_path(context)
        # Steps draw from streams addressed by chain, cycle, step and walker,
        # not from the state of the keyed context RNG:
        @test chain_path(context, ctx -> rand(get_rng(ctx), 100)) == reference
        @test chain_path(BATContext(rng = rootrng(5643))) == reference
        @test chain_path(BATContext(rng = rootrng(5644))) != reference

        # A context RNG without a stream key is the entropy of a keyed root,
        # so each chain built from it draws fresh streams:
        unkeyed = BATContext(rng = Philox4x((564, 3)))
        @test chain_path(unkeyed) != chain_path(unkeyed)
        @test chain_path(BATContext(rng = Philox4x((564, 3)))) == chain_path(BATContext(rng = Philox4x((564, 3))))
    end

    @testset "sampling runs" begin
        algorithm = TransformedMCMC(nchains = 2, nsteps = 500)
        context = BATContext(rng = Philox4x((564, 5)))
        samples1 = bat_sample(target, algorithm, context).result
        # A second run on the same context draws a fresh root:
        samples2 = bat_sample(target, algorithm, context).result
        @test samples1.v != samples2.v
        @test bat_sample(target, algorithm, BATContext(rng = Philox4x((564, 5)))).result.v == samples1.v
        # The chains of a run draw from different streams:
        chainid = samples1.info.chainid
        ids = unique(chainid)
        @test length(ids) == 2
        @test first(samples1.v[chainid .== ids[1]], 5) != first(samples1.v[chainid .== ids[2]], 5)

        keyed_context = BATContext(rng = rootrng(5645))
        samples3 = bat_sample(target, algorithm, keyed_context).result
        @test bat_sample(target, algorithm, keyed_context).result.v != samples3.v
        @test bat_sample(target, algorithm, BATContext(rng = rootrng(5645))).result.v == samples3.v
    end
end
