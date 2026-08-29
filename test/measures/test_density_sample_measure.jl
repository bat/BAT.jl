# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random
using StableRNGs: StableRNG
using LogarithmicNumbers: ULogarithmic
using DensityInterface, MeasureBase, ValueShapes
using Distributions, Statistics, StatsBase, IntervalSets
using MeasureBase: weightedmeasure

using BAT: DensitySampleMeasure, samplesof, empiricalof, getess

struct _ZeroFloatRNG <: AbstractRNG end
struct _ThirdFloatRNG <: AbstractRNG end

Random.rand(::_ZeroFloatRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float64},Float64}) = 0.0
Random.rand(::_ThirdFloatRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float64},Float64}) = 1 / 3

@testset "density_sample_measure" begin
    context = BATContext(rng = StableRNG(564001))
    fixture_rng = StableRNG(564002)
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
        rng = StableRNG(564003)
        x = rand(rng, dsm)
        @test x in smpls.v
        X = rand(rng, dsm^100)
        @test length(X) == 100
        @test all(in(smpls.v), X)

        @testset "zero-mass intervals" begin
            for weights in (Int[0, 1], Float32[0, 0, 1], BigFloat[1, 0])
                values = collect(eachindex(weights))
                point_mass = DensitySampleMeasure(DensitySampleVector(
                    v = values,
                    logd = zeros(length(weights)),
                    weight = weights,
                ))
                positive_idx = only(findall(>(0), weights))

                @test rand(_ZeroFloatRNG(), point_mass) == values[positive_idx]

                resampled = samplesof(evalmeasure(
                    point_mass,
                    RandResampling(nsamples = 4),
                    BATContext(rng = _ZeroFloatRNG()),
                ))
                @test resampled.v == fill(values[positive_idx], 4)
            end

            interior_boundary = DensitySampleMeasure(DensitySampleVector(
                v = [1, 2, 3],
                logd = zeros(3),
                weight = [1, 0, 2],
            ))
            @test rand(_ThirdFloatRNG(), interior_boundary) == 3
            resampled = samplesof(evalmeasure(
                interior_boundary,
                RandResampling(nsamples = 4),
                BATContext(rng = _ThirdFloatRNG()),
            ))
            @test resampled.v == fill(3, 4)

            # Defensive fallback if floating multiplication reaches total mass:
            @test BAT._weight_cdf_idx([1.0, 1.0], 1.0) == 1
        end
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
                v = [randn(fixture_rng, 2) for _ in 1:3], logd = zeros(3), weight = bad_w
            )
            @test_throws ArgumentError DensitySampleMeasure(bad_smpls)
        end
    end

    @testset "owned sampling weights" begin
        owned_info = [Dict(:origin => :input), Dict(:origin => :input)]
        owned_aux = [Dict(:source => 1), Dict(:source => 2)]
        owned_smpls = DensitySampleVector(
            v = [1.0, 2.0], logd = [-1.0, -2.0], weight = [1.0, 2.0],
            info = owned_info, aux = owned_aux,
        )
        owned_dsm = DensitySampleMeasure(owned_smpls)
        @test samplesof(owned_dsm) !== owned_smpls
        @test samplesof(owned_dsm).weight !== owned_smpls.weight
        @test samplesof(owned_dsm).v === owned_smpls.v
        @test samplesof(owned_dsm).logd === owned_smpls.logd
        @test samplesof(owned_dsm).info === owned_smpls.info
        @test samplesof(owned_dsm).aux === owned_smpls.aux

        # Sampling weights are a construction-time snapshot. Reconstruct
        # instead of mutating an already-constructed measure's owned weights:
        owned_smpls.weight .= [0.0, 5.0]
        @test samplesof(owned_dsm).weight == [1.0, 2.0]
        @test mean(owned_dsm) ≈ 5 / 3
        @test any(==(1.0), rand(StableRNG(564010), owned_dsm^20))

        reconstructed_dsm = DensitySampleMeasure(owned_smpls)
        @test samplesof(reconstructed_dsm).weight == [0.0, 5.0]
        @test all(==(2.0), rand(StableRNG(564004), reconstructed_dsm^20))

        # Canonical construction retains the existing numeric sampling law:
        for w_extreme in ([typemax(Int), typemax(Int), 4], [1e300, 2e300, 0.5e300])
            xdsm = DensitySampleMeasure(DensitySampleVector(
                v = [randn(fixture_rng, 2) for _ in 1:3], logd = zeros(3), weight = w_extreme
            ))
            @test rand(StableRNG(564005), xdsm) in samplesof(xdsm).v
        end

        # A tail below Float32 spacing remains visible in the owned CDF:
        tail_weight = eps(Float32) / 2
        tail_dsm = DensitySampleMeasure(DensitySampleVector(
            v = [1.0, 2.0], logd = zeros(2), weight = Float32[1, tail_weight]
        ))
        @test tail_dsm._cumulative_weight == [1.0, 1 + Float64(tail_weight)]

        scaled_dsm = DensitySampleMeasure(DensitySampleVector(
            v = [10.0, 20.0, 30.0], logd = zeros(3), weight = [2.0, 4.0, 0.0]
        ))
        rescaled_dsm = DensitySampleMeasure(DensitySampleVector(
            v = [10.0, 20.0, 30.0], logd = zeros(3), weight = [20.0, 40.0, 0.0]
        ))
        @test scaled_dsm._cumulative_weight == rescaled_dsm._cumulative_weight
        @test rand(StableRNG(564011), scaled_dsm^32) == rand(StableRNG(564011), rescaled_dsm^32)

        log_dsm = DensitySampleMeasure(DensitySampleVector(
            v = [1.0, 2.0], logd = zeros(2),
            weight = exp.(ULogarithmic, [0.0, log(2.0)]),
        ))
        @test log_dsm._cumulative_weight == [0.5, 1.5]

        source_weights = [2.0, 4.0, 0.0]
        source_view = view(source_weights, :)
        view_dsm = DensitySampleMeasure(DensitySampleVector(
            v = [1.0, 2.0, 3.0], logd = zeros(3), weight = source_view,
        ))
        @test samplesof(view_dsm).weight !== source_view
        @test view_dsm._cumulative_weight == [0.5, 1.5, 1.5]
        source_weights .= [0.0, 0.0, 5.0]
        @test rand(_ZeroFloatRNG(), view_dsm) == 1.0

        owner = DensitySampleMeasure(DensitySampleVector(
            v = [1.0, 2.0], logd = zeros(2), weight = [1.0, 2.0],
        ))
        divergent = DensitySampleMeasure(
            DensitySampleVector((
                samplesof(owner).v, samplesof(owner).logd, samplesof(owner).weight,
                samplesof(owner).info, samplesof(owner).aux,
            )),
            copy(owner._cumulative_weight), nothing, nothing, massof(owner),
        )
        repaired = BAT._with_sample_weights(divergent, owner)
        @test repaired !== divergent
        @test samplesof(repaired).weight === samplesof(owner).weight
        @test repaired._cumulative_weight === owner._cumulative_weight
    end

    @testset "resampling" begin
        em_rand = evalmeasure(dsm, RandResampling(nsamples = 500), context)
        rsmpls = samplesof(em_rand)
        @test length(rsmpls) == 500
        @test all(==(1), rsmpls.weight)
        @test all(in(smpls.v), rsmpls.v)
        @test getess(empiricalof(em_rand)) == n * 500 / (n + 500)

        identity_smpls = DensitySampleVector(
            v = [1.0, 2.0, 3.0], logd = [-1.0, -2.0, -3.0], weight = fill(2.0, 3)
        )
        identity_dsm = DensitySampleMeasure(identity_smpls, ess = 2.5, mass = 3.0)
        identity_out = empiricalof(evalmeasure(
            identity_dsm,
            SystematicResampling(nsamples = 3),
            BATContext(rng = StableRNG(1)),
        ))
        @test samplesof(identity_out).v == identity_smpls.v
        @test samplesof(identity_out).logd == identity_smpls.logd
        @test all(isone, samplesof(identity_out).weight)
        @test massof(identity_out) == massof(identity_dsm)
        @test getess(identity_out) == getess(identity_dsm)

        # Matching indices alone do not remove the resampling variance:
        nonuniform_smpls = DensitySampleVector(
            v = [1.0, 2.0], logd = [-1.0, -2.0], weight = [0.6, 0.4]
        )
        nonuniform_dsm = DensitySampleMeasure(nonuniform_smpls, ess = 2.0)
        nonuniform_out = empiricalof(evalmeasure(
            nonuniform_dsm,
            SystematicResampling(nsamples = 2),
            BATContext(rng = StableRNG(1)),
        ))
        @test samplesof(nonuniform_out).v == nonuniform_smpls.v
        @test getess(nonuniform_out) == 1.0

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
