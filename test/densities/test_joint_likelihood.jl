# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, Distributions

@testset "joint_likelihood" begin
    like1 = logfuncdensity(p -> logpdf(Normal(p.mu, p.sigma), 1.2))
    like2 = logfuncdensity(p -> logpdf(Normal(p.mu, 2 * p.sigma), 0.7))

    jlike = joint_likelihood(like1, like2)
    @test jlike isa BAT.JointLikelihood
    @test DensityKind(jlike) isa IsDensity

    p = (mu = 1.0, sigma = 0.5)
    @test @inferred(logdensityof(jlike, p)) ≈ logdensityof(like1, p) + logdensityof(like2, p)
    @test log(jlike(p)) ≈ logdensityof(jlike, p)

    @test joint_likelihood(like1).likelihoods == (like1,)
    @test_throws MethodError joint_likelihood()

    # Nested joint likelihoods flatten:
    like3 = logfuncdensity(p -> logpdf(Exponential(p.sigma), 0.4))
    @test joint_likelihood(jlike, like3).likelihoods isa Tuple{Vararg{Any,3}}

    # Components are converted like PosteriorMeasure likelihoods, so
    # transformed likelihoods and plain log-valued functions work as well:
    f_trafo = q -> (mu = q.m, sigma = q.s)
    q_t = (m = 1.0, s = 0.5)
    @test logdensityof(joint_likelihood(like1 ∘ f_trafo, like2 ∘ f_trafo), q_t) ≈ logdensityof(jlike, f_trafo(q_t))
    fl = p -> exp(BAT.ULogarithmic, logpdf(Normal(p.mu, p.sigma), 1.2))
    @test logdensityof(joint_likelihood(fl, like2), p) ≈ logdensityof(jlike, p)
    # Measure-like components are rejected:
    @test_throws ArgumentError joint_likelihood(Normal(), like2)

    # Precomposing a transform distributes over the components:
    g = q -> (mu = q.m_offs + 1.0, sigma = q.s)
    q = (m_offs = 0.0, s = 0.5)
    jlike_g = BAT._precompose_density(jlike, g)
    @test jlike_g isa BAT.JointLikelihood
    @test logdensityof(jlike_g, q) ≈ logdensityof(jlike, g(q))
end
