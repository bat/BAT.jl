# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Distributions, StatsBase
using DensityInterface: logfuncdensity
using MeasureBase: UnknownMass, massof, weightedmeasure
using Random123: Philox4x


@testset "importance_samplers" begin
    function test_moments(dist::Distribution, algo::BAT.AbstractSamplingAlgorithm; rtol::Real=0.01)
        # ToDo: Wrap in @inferred when type stable
        samples = bat_sample(dist, algo).result

        @test isapprox(@inferred(mean(samples)), @inferred(mean(dist)), rtol=rtol)
        @test isapprox(@inferred(var(samples)), @inferred(var(dist)), rtol=rtol)
    end

    @testset "sobol_sampler" begin
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])
        algo = SobolSampler(nsamples=10^5)

        test_moments(dist, algo)
    end

    @testset "grid_sampler" begin
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])
        algo = GridSampler(ppa=500)

        # ToDo: Find a better way than using a huge rtol:
        test_moments(dist, algo, rtol=0.5)
    end

    @testset "prior_importance_sampler prior mass" begin
        likelihood_scale = 2.0
        prior_scales = (1.0, 0.25, 7.0, 16.0)
        nsamples = 32
        results = map(prior_scales) do prior_scale
            prior = weightedmeasure(log(prior_scale), Normal())
            posterior = PosteriorMeasure(logfuncdensity(_ -> log(likelihood_scale)), prior)
            context = BATContext(rng = Philox4x((564, 2902)))
            em = evalmeasure(posterior, PriorImportanceSampler(nsamples = nsamples), context)
            (;em, samples = BAT.samplesof(em))
        end
        reference = first(results)

        for (prior_scale, result) in zip(prior_scales, results)
            @test Float64(massof(result.em)) ≈ prior_scale * likelihood_scale rtol = eps(Float32)
            @test log(massof(result.em)) isa Float64
            @test result.samples.v == reference.samples.v
            @test result.samples.weight ≈ fill(likelihood_scale, nsamples)
            @test BAT.getess(result.em) == nsamples
        end

        unknown_prior = EvaluatedMeasure(Normal(), mass = UnknownMass())
        unknown_posterior = PosteriorMeasure(
            logfuncdensity(_ -> log(likelihood_scale)),
            unknown_prior,
        )
        unknown_em = evalmeasure(
            unknown_posterior,
            PriorImportanceSampler(nsamples = nsamples),
            BATContext(rng = Philox4x((564, 2903))),
        )
        @test massof(unknown_em) isa UnknownMass
    end
end
