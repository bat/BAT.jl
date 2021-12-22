# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, StatsBase, Distributions, DensityInterface, ValueShapes
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

    posterior = PosteriorDensity(likelihood, prior)
    algorithm = EllipsoidalNestedSampling(max_ncalls = 10^5)
    r = BAT.sample_and_verify(posterior, algorithm, dist)
    @test r.verified

    smpls = r.result
    @test logdensityof(posterior).(smpls.v) ≈ smpls.logd

    logz_expected = -log(prod(maximum.(prior.a.v) .- minimum.(prior.a.v)))
    @test isapprox(r.logintegral.val, logz_expected, atol = 100 * r.logintegral.err)

    @test r.ess > 50
end
