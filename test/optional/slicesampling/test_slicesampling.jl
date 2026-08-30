# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using DensityInterface
using Distributions
using Random
using Statistics
using Test
using ValueShapes

import SliceSampling

prior = Normal(-0.5, 1.2)
observation, observation_std = 1.25, 0.5
likelihood = logfuncdensity(x -> logpdf(Normal(x, observation_std), observation))
posterior = PosteriorMeasure(likelihood, prior)
algorithm = SliceMCMCSampling(nsamples = 128, n_burnin = 32)
em = evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(0x534c494345)))
smpls = samplesof(em)
posterior_var = inv(inv(var(prior)) + inv(observation_std^2))
posterior_mean = posterior_var * (mean(prior) / var(prior) + observation / observation_std^2)

@test mean(smpls) ≈ posterior_mean atol = 0.12

prior = NamedTupleDist(a = Normal(1.5, 0.75), b = Normal(-2.0, 0.5))
posterior = PosteriorMeasure(logfuncdensity(_ -> 0.0), prior)
sampler = SliceSampling.RandPermGibbs(SliceSampling.SliceDoublingOut(1.0))
algorithm = SliceMCMCSampling(sampler = sampler, nsamples = 64, n_burnin = 16)
smpls = samplesof(evalmeasure(posterior, algorithm, BATContext(rng = Xoshiro(0x5348415045))))

@test mean(getproperty.(smpls.v, :b)) ≈ mean(prior.b) atol = 0.15
