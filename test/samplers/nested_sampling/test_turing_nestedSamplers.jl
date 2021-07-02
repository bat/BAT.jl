# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, StatsBase, Distributions
using HypothesisTests

import NestedSamplers


@testset "test_turing_nestedSamplers" begin

    p = Uniform(-80,80)
    prior = BAT.NamedTupleDist(a=[p,p,p])

    dist = [
        MixtureModel(Normal, [(-50.0,2.5),(50.0,2.5)],[0.2,0.8]),
        MixtureModel(Normal, [(-20.0,2.5),(20.0,2.5)],[0.7,0.3]),
        MixtureModel(Normal, [(-40.0,5.0),(10.0,1.0)],[0.5,0.5])
    ]

    likelihood = params -> begin
        r1 = logpdf(dist[1],params.a[1])
        r2 = logpdf(dist[2],params.a[2])
        r3 = logpdf(dist[3],params.a[3])
        return LogDVal(r1+r2+r3)
    end

    posterior = PosteriorDensity(likelihood, prior)
    algorithm = TuringNestedSamplers()
    r = bat_sample(posterior, algorithm);
    smpls = r.result

    @test logvalof(posterior).(smpls.v) ≈ smpls.logd

    iid = BAT.NamedTupleDist(a=dist)
    iidsamples, chains = bat_sample(iid, IIDSampling());
    @test ones(3) ≈ isapprox.(bat_compare(smpls,iidsamples).result.ks_p_values, 1.0, atol = 0.3)

    logz_expected = -log(prod(160))*3
    @test isapprox(r.logintegral.val, logz_expected, atol = 100 * r.logintegral.err)

end