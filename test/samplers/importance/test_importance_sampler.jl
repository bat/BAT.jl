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
        algorithm = PriorImportanceSampler(nsamples = 32)
        likelihood = logfuncdensity(_ -> log(2.0))
        posterior = PosteriorMeasure(
            likelihood,
            weightedmeasure(log(7.0), Normal()),
        )
        result = evalmeasure(
            posterior,
            algorithm,
            BATContext(rng = Philox4x((564, 2902))),
        )
        @test Float64(massof(result)) ≈ 14.0 rtol = 1e-6

        unknown = evalmeasure(
            PosteriorMeasure(likelihood, EvaluatedMeasure(Normal(), mass = UnknownMass())),
            algorithm,
            BATContext(rng = Philox4x((564, 2903))),
        )
        @test massof(unknown) isa UnknownMass
    end
end
