# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Distributions, Statistics
using LazyReports: LazyReport, lazyreport


@testset "evaluated_measure" begin
    context = BATContext()
    dist = MvNormal([0.4, 0.6], [2.0 1.2; 1.2 3.0])
    m = batmeasure(dist)
    n = 2

    smpls = bat_sample(m, IIDSampling(nsamples = 10^4), context).result

    em_with_samples = EvaluatedMeasure(m, samples = smpls)
    em_without_samples = EvaluatedMeasure(m)

    # Absent optional content is `missing`, not `nothing` - guards that test for
    # `nothing` silently take the wrong branch:
    @test BAT.maybe_samplesof(em_without_samples) === missing
    @test BAT.maybe_samplesof(em_with_samples) === smpls

    @testset "approximate mean and cov" begin
        @test BAT._approx_mean(em_with_samples, n) == mean(smpls)
        @test BAT._approx_cov(em_with_samples, n) == cov(smpls)

        # Without samples both must fall back to the underlying measure:
        @test BAT._approx_mean(em_without_samples, n) == BAT._approx_mean(m, n)
        @test BAT._approx_cov(em_without_samples, n) == BAT._approx_cov(m, n)
    end

    @testset "estimated max logd" begin
        @test BAT._estimated_max_logd(em_with_samples) == BAT._estimated_max_logd(smpls)
        @test BAT._estimated_max_logd(em_without_samples) === BAT._estimated_max_logd(m)
    end

    @testset "report generation" begin
        @test lazyreport(em_with_samples) isa LazyReport
        @test lazyreport(em_without_samples) isa LazyReport
    end
end
