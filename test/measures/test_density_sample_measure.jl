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
        # Reweighting shifts the sample log-densities, variates and weights
        # are shared:
        @test samplesof(wdsm).v === samplesof(dsm).v
        @test samplesof(wdsm).weight === samplesof(dsm).weight
        @test samplesof(wdsm).logd ≈ samplesof(dsm).logd .+ 1.3
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

    @testset "weight invariants" begin
        # Empirical-measure weights must be finite and non-negative (a
        # violation would make the subsampling CDF non-monotone) and not
        # all zero (nothing to draw):
        for bad_w in ([1.0, -1.0, 2.0], [1.0, NaN, 2.0], [1.0, Inf, 2.0], [0.0, 0.0, 0.0])
            bad_smpls = DensitySampleVector(
                v = [randn(2) for _ in 1:3], logd = zeros(3), weight = bad_w
            )
            @test_throws ArgumentError DensitySampleMeasure(bad_smpls)
        end
    end

    @testset "live weights" begin
        # The sample vector is shared, not copied: draw probabilities and
        # statistics must stay coherent with the weights as the caller
        # sees (and possibly mutates) them:
        lw_smpls = DensitySampleVector(
            v = [randn(2) for _ in 1:4], logd = zeros(4), weight = [1.0, 1.0, 1.0, 1.0]
        )
        lw_dsm = DensitySampleMeasure(lw_smpls)
        @test samplesof(lw_dsm) === lw_smpls
        samplesof(lw_dsm).weight .= [0.0, 0.0, 5.0, 0.0]
        rng = Random.default_rng()
        @test all(x -> x == lw_smpls.v[3], [rand(rng, lw_dsm) for _ in 1:20])
        @test mean(lw_dsm) ≈ lw_smpls.v[3]
        # Invalid mutated weights are caught at draw time:
        samplesof(lw_dsm).weight .= [0.0, 0.0, 0.0, 0.0]
        @test_throws ArgumentError rand(rng, lw_dsm)

        # Extreme weight scales are safe - the subsampling CDF is built
        # from canonical relative weights:
        for w_extreme in ([typemax(Int), typemax(Int), 4], [1e300, 2e300, 0.5e300])
            xdsm = DensitySampleMeasure(DensitySampleVector(
                v = [randn(2) for _ in 1:3], logd = zeros(3), weight = w_extreme
            ))
            @test rand(rng, xdsm) in samplesof(xdsm).v
        end
    end

    @testset "resampling" begin
        em_rand = evalmeasure(dsm, RandResampling(nsamples = 500), context)
        rsmpls = samplesof(em_rand)
        @test length(rsmpls) == 500
        @test all(==(1), rsmpls.weight)
        @test all(in(smpls.v), rsmpls.v)
        @test getess(empiricalof(em_rand)) <= getess(dsm)

        # Order-preserving systematic resampling keeps MCMC sample-id
        # provenance, multinomial resampling destroys the process order
        # and clears it:
        ids = [BAT.MCMCSampleID(Int32(1), Int32(1), Int32(1), Int64(i), Int32(1), true) for i in eachindex(smpls)]
        smpls_tagged = DensitySampleVector(v = smpls.v, logd = smpls.logd, weight = smpls.weight, info = ids)
        dsm_tagged = DensitySampleMeasure(smpls_tagged)
        r_sys = samplesof(evalmeasure(dsm_tagged, SystematicResampling(nsamples = 100), context))
        @test eltype(r_sys.info) <: BAT.MCMCSampleID
        r_rand = samplesof(evalmeasure(dsm_tagged, RandResampling(nsamples = 100), context))
        @test eltype(r_rand.info) === Nothing

        em_ord = evalmeasure(dsm, SystematicResampling(nsamples = 500), context)
        osmpls = samplesof(em_ord)
        @test all(==(1), osmpls.weight)
        @test all(in(smpls.v), osmpls.v)
        @test issorted(indexin(osmpls.v, smpls.v))
    end
end
