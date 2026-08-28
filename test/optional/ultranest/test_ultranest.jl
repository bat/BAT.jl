# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using Random, StatsBase, Distributions
using DensityInterface
using MeasureBase: massof
using PythonCall: pyimport

import UltraNest


@testset "test_ultranest" begin
    pyimport("numpy").random.seed(564)
    context = BATContext(rng = Xoshiro(564084))

    dist = product_distribution([
        MixtureModel([truncated(Normal(-1, 0.1), -2, 0), truncated(Normal(1, 0.1), 0, 2)], [0.5, 0.5]),
        MixtureModel([truncated(Normal(-2, 0.25), -3, -1), truncated(Normal(2, 0.25), 1, 3)], [0.3, 0.7]),
        MixtureModel([truncated(Normal(-5, 0.25), -6, -4), truncated(Normal(5, 0.25), 4, 6)], [0.2, 0.8]),
    ])

    prior = product_distribution(Uniform.(minimum.(dist.v), maximum.(dist.v)))

    likelihood = let dist = dist
        logfuncdensity(function (v::AbstractVector{<:Real})
            ll = logpdf(dist, v)
            # lofpdf on MixtureModel returns NaN in gaps between distributions, and UltraNest
            # doesn't like -Inf, so return -1E10
            T = promote_type(Float32, typeof(ll))
            # isnan(ll) here only required for Distributions < v0.25
            isnan(ll) || isinf(ll) && ll < 0 ? T(-1E10) : T(ll)
        end)
    end

    posterior = PosteriorMeasure(likelihood, prior)
    algorithm = ReactiveNestedSampling(show_status = false)
    # Nested sampling produces importance-weighted samples, for which the
    # autocorrelation-based ESS is not meaningful:
    r = BAT.sample_and_verify(posterior, algorithm, dist, context, essalg = KishESS(), max_retries = 0)
    @test r.verified
    @test r.n_retries == 0

    smpls = r.result
    @test logdensityof(posterior).(smpls.v) ≈ smpls.logd

    em = r.evaluated
    @test em isa EvaluatedMeasure
    @test BAT.validate_evalmeasure(em, context = context) === em

    # The equal-weight sample variant is algorithm-specific and lives in
    # the evaluation info:
    uwsmpls = BAT.evalinfo(em).result.uwresult
    @test logdensityof(posterior).(uwsmpls.v) ≈ uwsmpls.logd
    @test all(isequal(1), uwsmpls.weight)

    logz_expected = -log(prod(maximum.(prior.v) .- minimum.(prior.v)))
    logmass = log(massof(em))
    @test isapprox(logmass.val, logz_expected, atol = 10 * logmass.err)

    # Ultranest uses Kish's ESS estimator:
    ess = BAT.getess(BAT.empiricalof(em))
    @test ess ≈ bat_eff_sample_size(r.result, KishESS(), context).result

    @test ess > 50
end
