# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, StatsBase, Distributions, DensityInterface, ValueShapes
using MeasureBase: massof
using HypothesisTests

import NestedSamplers

@testset "test_ellipsoidal_nested_sampling" begin
    p = Uniform(-80,80)
    prior = BAT.NamedTupleDist(a=[p,p,p])

    dists = [
        MixtureModel(Normal, [(-50.0,2.5),(50.0,2.5)],[0.2,0.8]),
        MixtureModel(Normal, [(-20.0,2.5),(20.0,2.5)],[0.7,0.3]),
        MixtureModel(Normal, [(-40.0,5.0),(10.0,1.0)],[0.5,0.5])
    ]

    dist = NamedTupleDist(a = product_distribution(dists))

    likelihood = logfuncdensity(logdensityof(dist))

    posterior = PosteriorMeasure(likelihood, prior)
    algorithm = EllipsoidalNestedSampling(max_ncalls = 10^5)
    # Nested-sampling output is importance-weighted, so the sample
    # comparison must use Kish's ESS, not the autocorrelation-based one:
    r = BAT.sample_and_verify(posterior, algorithm, dist, essalg = KishESS())
    @test r.verified

    smpls = r.result
    @test logdensityof(posterior).(smpls.v) ≈ smpls.logd

    em = r.evaluated
    @test em isa EvaluatedMeasure
    @test BAT.validate_evalmeasure(em) === em

    logz_expected = -log(prod(maximum.(prior.a.v) .- minimum.(prior.a.v)))
    logintegral = log(massof(em))
    @test isapprox(logintegral.val, logz_expected, atol = 100 * logintegral.err)

    @test BAT.getess(BAT.empiricalof(em)) > 50
end
