# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using StableRNGs: StableRNG
using DensityInterface, MeasureBase, ValueShapes
using Distributions, Statistics
using MeasureBase: weightedmeasure

using BAT: DensitySampleMeasure, samplesof, empiricalof, getess

struct _ThirdFloatRNG <: AbstractRNG end

Random.rand(::_ThirdFloatRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float64},Float64}) = 1 / 3

@testset "density_sample_measure" begin
    context = BATContext(rng = StableRNG(564001))
    dist = NamedTupleDist(a = Normal(2, 1), b = Weibull())
    m = batmeasure(dist)

    n = 100
    smpls = BAT.samplesof(evalmeasure(m, IIDSampling(nsamples = n), context))

    dsm = DensitySampleMeasure(smpls, dof = getdof(m), ess = n)
    @testset "conversion" begin
        @test samplesof(batmeasure(smpls)) == smpls
        @test DensitySampleVector(dsm) == smpls
    end

    @testset "properties" begin
        @test getdof(dsm) == getdof(m)
        @test getess(dsm) == n
        @test massof(dsm) ≈ 1
        @test varshape(dsm) == varshape(smpls)
    end

    @testset "rand" begin
        rng = StableRNG(564003)
        x = rand(rng, dsm)
        @test x in smpls.v
        X = rand(rng, dsm^20)
        @test length(X) == 20
        @test all(in(smpls.v), X)

        boundary = DensitySampleMeasure(DensitySampleVector(
            v = [1, 2, 3], logd = zeros(3), weight = [1, 0, 2],
        ))
        @test rand(_ThirdFloatRNG(), boundary) == 3
        resampled = samplesof(evalmeasure(
            boundary, RandResampling(nsamples = 4), BATContext(rng = _ThirdFloatRNG()),
        ))
        @test resampled.v == fill(3, 4)
    end

    @testset "statistics" begin
        @test mean(dsm) == mean(smpls)
    end

    @testset "weightedmeasure" begin
        wdsm = weightedmeasure(1.3, dsm)
        # Reweighting shifts the sample log-densities and mass:
        @test samplesof(wdsm).logd ≈ samplesof(dsm).logd .+ 1.3
        @test massof(wdsm) ≈ exp(1.3)
    end

    @testset "owned sampling weights" begin
        owned_smpls = DensitySampleVector(
            v = [1.0, 2.0], logd = [-1.0, -2.0], weight = [1.0, 2.0],
        )
        owned_dsm = DensitySampleMeasure(owned_smpls)
        owned_smpls.weight .= [0.0, 5.0]
        @test samplesof(owned_dsm).weight == [1.0, 2.0]
    end

    @testset "resampling" begin
        em_rand = evalmeasure(dsm, RandResampling(nsamples = 500), context)
        rsmpls = samplesof(em_rand)
        @test length(rsmpls) == 500
        @test all(==(1), rsmpls.weight)
        @test all(in(smpls.v), rsmpls.v)
        @test getess(empiricalof(em_rand)) <= getess(dsm)
    end
end
