# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using DensityInterface
using Distributions
using Random
using Statistics
using Test
using ValueShapes

import EllipticalSliceSampling

prior = Normal(-0.5, 1.2)
observation, observation_std = 1.25, 0.5
posterior = PosteriorMeasure(
    logfuncdensity(x -> logpdf(Normal(x, observation_std), observation)), prior,
)
smpls = samplesof(evalmeasure(
    posterior,
    EllipticalSliceMCMCSampling(nsamples = 256, n_burnin = 32),
    BATContext(rng = Xoshiro(0x454c4c49505345)),
))
posterior_var = inv(inv(var(prior)) + inv(observation_std^2))
posterior_mean =
    posterior_var * (mean(prior) / var(prior) + observation / observation_std^2)
@test mean(smpls) ≈ posterior_mean atol = 0.06

prior = NamedTupleDist(a = Uniform(-2.0, 4.0), b = LogNormal(0.2, 0.4))
smpls = samplesof(evalmeasure(
    PosteriorMeasure(logfuncdensity(_ -> 0.0), prior),
    EllipticalSliceMCMCSampling(nsamples = 4, n_burnin = 1),
    BATContext(rng = Xoshiro(0x5052494f52)),
))
@test all(s -> -2 <= s.a <= 4 && s.b > 0, smpls.v)
