# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using DensityInterface, MeasureBase, ValueShapes
using Distributions, Statistics, StatsBase, IntervalSets
using MeasureBase: weightedmeasure

using BAT: DensitySampleMeasure, samplesof, empiricalof, getess

@testset "density_sample_measure" begin
    context = BATContext()
    dist = NamedTupleDist(a = Normal(2, 1), b = Weibull())
    m = batmeasure(dist)

    n = 10^4
    smpls = BAT.samplesof(evalmeasure(m, IIDSampling(nsamples = n), context))

    dsm = @inferred(DensitySampleMeasure(smpls, dof = getdof(m), ess = n))
    @test dsm isa DensitySampleMeasure

    @testset "conversion" begin
        @test convert(DensitySampleMeasure, smpls) == DensitySampleMeasure(smpls)
        @test batmeasure(smpls) isa DensitySampleMeasure
        @test samplesof(batmeasure(smpls)) == smpls
        @test DensitySampleVector(dsm) == smpls
        @test convert(DensitySampleVector, dsm) == smpls
    end

    @testset "properties" begin
        @test @inferred(samplesof(dsm)) === dsm._smpls
        @test @inferred(empiricalof(dsm)) === dsm
        @test @inferred(getdof(dsm)) == getdof(m)
        @test getess(dsm) == n
        @test massof(dsm) ≈ 1
        @test getdof(DensitySampleMeasure(smpls)) isa MeasureBase.NoDOF
        @test isnothing(getess(DensitySampleMeasure(smpls)))
        @test @inferred(varshape(dsm)) == varshape(smpls)
        @test BAT.supports_rand(dsm)
        @test MeasureBase.testvalue(dsm) == first(smpls.v)
        @test_throws ArgumentError logdensityof(dsm, first(smpls.v))
    end

    @testset "rand" begin
        rng = Random.default_rng()
        x = rand(rng, dsm)
        @test x in smpls.v
        X = rand(rng, dsm^100)
        @test length(X) == 100
        @test all(in(smpls.v), X)
    end

    @testset "statistics" begin
        @test mean(dsm) == mean(smpls)
        @test var(dsm) == var(smpls)
        udsm = unshaped(dsm, varshape(dsm))
        @test cov(udsm) == cov(unshaped.(smpls))
    end

    @testset "weightedmeasure" begin
        wdsm = weightedmeasure(1.3, dsm)
        @test samplesof(wdsm) === samplesof(dsm)
        @test massof(wdsm) ≈ exp(1.3)
        @test getess(wdsm) == getess(dsm)
    end

    @testset "unshaped" begin
        udsm = unshaped(dsm, varshape(dsm))
        @test samplesof(udsm) == unshaped.(smpls)
        @test getdof(udsm) == getdof(dsm)
        @test massof(udsm) == massof(dsm)
    end

    @testset "empty samples" begin
        empty_dsm = DensitySampleMeasure(smpls[1:0])
        @test length(samplesof(empty_dsm)) == 0
    end

    @testset "resampling" begin
        em_rand = evalmeasure(dsm, RandResampling(nsamples = 500), context)
        rsmpls = samplesof(em_rand)
        @test length(rsmpls) == 500
        @test all(==(1), rsmpls.weight)
        @test all(in(smpls.v), rsmpls.v)
        @test getess(empiricalof(em_rand)) <= getess(dsm)

        em_ord = evalmeasure(dsm, OrderedResampling(nsamples = 500), context)
        osmpls = samplesof(em_ord)
        @test all(==(1), osmpls.weight)
        @test all(in(smpls.v), osmpls.v)
        @test issorted(indexin(osmpls.v, smpls.v))
    end
end
