# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, Distributions, StatsBase
using FillArrays: Fill
using LogarithmicNumbers: ULogarithmic
using StableRNGs: StableRNG


@testset "bat_sample" begin
    context = BATContext()

    @testset "IIDSampling" begin
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])

        @test length(@inferred(bat_sample(dist, IIDSampling(nsamples = 10^3), context)).result) == 10^3

        @test @inferred(bat_sample(dist, context)).result isa DensitySampleVector
        @test bat_sample(dist, BAT.IIDSampling()).result isa DensitySampleVector
        @test @inferred(bat_sample(dist, BAT.IIDSampling(), context)).result isa DensitySampleVector

        samples = @inferred(bat_sample(dist, IIDSampling(nsamples = 10^5), context)).result
        @test maximum(BAT.dist_samples_mean_zscores(dist, samples, context)) < 5
        @test isapprox(cov(samples.v), [2.0 1.2; 1.2 3.0]; rtol = 0.05)
        @test all(isequal(1), samples.weight)

        dist_bmode = @inferred(bat_findmode(dist, context)).result
        @test @inferred(length(dist_bmode)) == 2

        dist_sample_vector_bmode = @inferred(bat_findmode(samples, context)).result
        @test @inferred(length(dist_sample_vector_bmode)) == 2

        @test isapprox(var(bat_sample(Normal(), BAT.IIDSampling(nsamples = 10^3), context).result), 1, rtol = 0.25)
    end

    @testset "RandResampling" begin
        dist = Normal()
        result = @inferred(bat_sample(dist, IIDSampling(nsamples = 2), context)).result #Draw to samples from Normal dist

        @test @inferred(bat_sample(result, context)).result isa DensitySampleVector#Check data types 
        @test bat_sample(result, RandResampling(nsamples = 100)).result isa DensitySampleVector
        @test @inferred(bat_sample(result, BAT.RandResampling(), context)).result isa DensitySampleVector

        samples_rdm = @inferred(bat_sample(result, RandResampling(nsamples = 10^5), context)).result #Sample 100 times from the 2-sample space
        @test length(@inferred(bat_sample(result, RandResampling(nsamples = 100), context)).result) == 100#Check shape is ok
        @test sort(unique(samples_rdm.v)) == sort(result.v)#check it only samples from the 2-sample space
        # Means should agree up to the resampling noise, which scales with the spread of the base samples:
        @test isapprox(mean(samples_rdm), mean(result), atol = 0.01 * abs(result.v[1] - result.v[2]))

        immutable_weighted_samples = DensitySampleVector(
            [[1.0], [2.0]],
            zeros(2),
            weight = Fill(2.0, 2),
        )
        @test length(bat_sample(immutable_weighted_samples, RandResampling(nsamples = 1), context).result) == 1

        empty_samples = DensitySampleVector(Vector{Vector{Float64}}(), Float64[])
        @test isempty(bat_sample(empty_samples, RandResampling(nsamples = 0), context).result)

        # All-zero weights carry no distribution to resample from, they
        # are rejected when the samples become an empirical measure:
        zero_weight_samples = DensitySampleVector([[1.0], [2.0]], zeros(2), weight = zeros(2))
        @test_throws ArgumentError bat_sample(zero_weight_samples, RandResampling(nsamples = 0), context)

        n = 10^4
        logweights = -1000.0 .- (0:49)
        weighted_samples = DensitySampleVector(
            [[Float64(i)] for i in eachindex(logweights)],
            zeros(length(logweights)),
            weight = exp.(ULogarithmic, logweights),
        )
        resamples = bat_sample(
            weighted_samples,
            RandResampling(nsamples = n),
            BATContext(rng = StableRNG(892374)),
        ).result
        expected_fraction = inv(sum(exp.(-(0:49))))

        @test count(==(1.0), only.(resamples.v)) / n ≈ expected_fraction atol = 0.02
        @test all(isone, resamples.weight)
    end

    @testset "SystematicResampling" begin
        dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])
        result = @inferred(bat_sample(dist, IIDSampling(nsamples = 10^5), context)).result

        # Systematic resampling yields exactly the requested number of samples:
        @test length(@inferred(bat_sample(result, SystematicResampling(nsamples = 10), context)).result.v) == 10

        @test @inferred(bat_sample(result, context)).result isa DensitySampleVector#Check that types are consistent
        @test @inferred(bat_sample(result, BAT.SystematicResampling(), context)).result isa DensitySampleVector
        @test bat_sample(result, BAT.SystematicResampling()).result isa DensitySampleVector

        resamples = @inferred(bat_sample(result, SystematicResampling(nsamples = length(result)), context)).result
        @test result == resamples

        # Wide accumulation preserves a tail below Float32 spacing:
        tail_weight = eps(Float32) / 2
        tail_samples = DensitySampleVector(
            v = [1, 2], logd = zeros(2), weight = Float32[1, tail_weight]
        )
        tail_resamples = bat_sample(
            tail_samples,
            SystematicResampling(nsamples = 10_000),
            BATContext(rng = StableRNG(1499)),
        ).result
        @test count(==(2), tail_resamples.v) == 1

        # The old name remains as a deprecated alias:
        @test BAT.OrderedResampling === SystematicResampling
    end
end


import Measurements
using MeasureBase: massof
using DensityInterface: logfuncdensity
using Distributions: Normal, logpdf

@testset "transformed space preservation" begin
    context = BATContext()
    post = PosteriorMeasure(logfuncdensity(v -> logpdf(Normal(1.0, 0.5), v.a)), distprod(a = Normal(0, 3)))
    em = evalmeasure(post, TransformedMCMC(nchains = 2, nsteps = 400), context)

    @test em.empirical isa BAT.BispacedMeasure
    @test em.transform_intent === NormalBased()
    @test !isnothing(em.empirical.transformed)
    @test !isnothing(em.unevaluated.transformed)

    @test BAT.empiricalof(em) === em.empirical.main

    # Re-entering the same space preserves measure and transform-function
    # identity and needs no sample transport:
    m_z, f_z = BAT.transform_and_unshape(NormalBased(), em, context)
    @test BAT.unevaluated(m_z) === em.unevaluated.transformed
    @test f_z === em.f_transform
    @test BAT.empiricalof(m_z) === em.empirical.transformed

    # A different intent falls back to the regular path:
    m_u, _ = BAT.transform_and_unshape(UniformBased(), em, context)
    @test BAT.unevaluated(m_u) !== em.unevaluated.transformed

    # The full transformed-space-view contract holds:
    @test BAT.validate_evalmeasure(em, context = context) === em

    # Follow-up evaluations work on the enriched measure:
    em2 = evalmeasure(em, BridgeSampling(), context)
    # Masses are stored on the canonical logarithmic scale:
    @test massof(em2) isa BAT.ULogarithmic
    @test isfinite(Measurements.value(log(massof(em2))))
    @test em2.empirical === em.empirical
    @test em2.unevaluated === em.unevaluated
end
