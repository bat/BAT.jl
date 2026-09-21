# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, Distributions, ValueShapes
using ChangesOfVariables: with_logabsdet_jacobian
using InverseFunctions: inverse
using MeasureBase: logdensities

using BAT: FullMeasureTransform, PriorSubstitution, getprior

@testset "transform_algorithms" begin
    context = BATContext()
    posterior = BAT.example_posterior()
    xs = bat_sample(getprior(posterior), IIDSampling(nsamples = 10), context).result.v

    # Transformed measures carry the density of the original measure, up to
    # the log-abs-det-Jacobian of the transformation:
    for algorithm in (PriorSubstitution(), FullMeasureTransform())
        for intent in (NormalBased(), UniformBased())
            tr = bat_transform(intent, posterior, algorithm, context)
            for x in xs
                y, ladj = with_logabsdet_jacobian(tr.f_transform, x)
                @test logdensityof(tr.result, y) ≈ logdensityof(posterior, x) - ladj
            end
        end
    end

    # Prior substitution keeps the likelihood's forward model accessible, with
    # the map into the original space composed in (BATMGVIExt extracts it):
    tr_ps = bat_transform(NormalBased(), posterior, PriorSubstitution(), context)
    likelihood, likelihood_ps = BAT.getlikelihood(posterior), BAT.getlikelihood(tr_ps.result)
    obs = BAT._get_observation(likelihood)
    @test BAT._get_observation(likelihood_ps) == obs
    @test logpdf(BAT._get_model(likelihood_ps)(tr_ps.f_transform(first(xs))), obs) ≈
        logpdf(BAT._get_model(likelihood)(first(xs)), obs)

    # The likelihood is never evaluated outside of the prior's support, neither
    # directly nor after the prior has been substituted (the boundary of the
    # original support transports to infinite variates):
    strict_likelihood = logfuncdensity(function (x)
        isfinite(x) || error("likelihood evaluated at non-finite variate $x")
        -x^2
    end)
    exp_posterior = PosteriorMeasure(strict_likelihood, Exponential())

    @test logdensityof(exp_posterior, -1.0) == -Inf
    @test logdensities(exp_posterior, [-1.0, 1.0]) == [-Inf, logdensityof(exp_posterior, 1.0)]

    tr_exp = bat_transform(NormalBased(), exp_posterior, context)
    # Finite inputs give finite variates, the likelihood is evaluated:
    @test isfinite(logdensityof(tr_exp.result, [40.0]))
    # Infinite variates never reach the likelihood:
    @test logdensityof(tr_exp.result, [Inf]) == -Inf
    x_exp, ladj_exp = with_logabsdet_jacobian(inverse(tr_exp.f_transform), [0.5])
    @test logdensityof(tr_exp.result, [0.5]) ≈ logdensityof(exp_posterior, x_exp) + ladj_exp
end
