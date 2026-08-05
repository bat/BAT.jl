# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, Distributions, FunctionChains, ValueShapes
import ForwardDiff, Zygote

@testset "posterior_measure" begin
    prior = distprod(m_offs = Uniform(-2, 2), s = Uniform(0.1, 2))
    ℒ = logfuncdensity(p -> logpdf(Normal(p.mu, p.sigma), 1.2))
    reparam = q -> (mu = q.m_offs + 1.0, sigma = q.s)
    q = (m_offs = 0.0, s = 0.5)

    @testset "transformed likelihoods" begin
        # A function chain or composed function that ends in a density is a
        # transformed likelihood, the transform gets precomposed:
        for f_likelihood in (fchain(reparam, ℒ), ℒ ∘ reparam)
            ℒ_split, g = BAT._split_density_transform(f_likelihood)
            @test ℒ_split === ℒ
            @test g(q) == reparam(q)
            @test logdensityof(BAT.getlikelihood(PosteriorMeasure(f_likelihood, prior)), q) ≈ logdensityof(ℒ, reparam(q))
        end

        scale = q -> (m_offs = 2 * q.m_scaled, s = q.s)
        q_scaled = (m_scaled = 0.0, s = 0.5)
        ℒ_nested = BAT.getlikelihood(PosteriorMeasure(ℒ ∘ reparam ∘ scale, prior))
        @test logdensityof(ℒ_nested, q_scaled) ≈ logdensityof(ℒ, reparam(scale(q_scaled)))

        # Precomposition keeps the structure of the likelihood intact:
        jlike = joint_likelihood(ℒ, logfuncdensity(p -> logpdf(Exponential(p.sigma), 0.4)))
        ℒ_joint = BAT.getlikelihood(PosteriorMeasure(fchain(reparam, jlike), prior))
        @test ℒ_joint isa BAT.JointLikelihood
        @test logdensityof(ℒ_joint, q) ≈ logdensityof(jlike, reparam(q))

        # Functions that don't end in a density are evaluated via logvalof:
        f_dval = q -> jlike(reparam(q))
        @test BAT._split_density_transform(f_dval) === nothing
        @test logdensityof(BAT.getlikelihood(PosteriorMeasure(f_dval, prior)), q) ≈ logdensityof(jlike, reparam(q))
    end

    @testset "AD through measure construction" begin
        # Regression test: constructing the measure inside the differentiated
        # function must not break Zygote (its pullback for splatted function
        # composition, as used by ffcomp, is defective):
        x = [0.3, 0.6]
        f_logd = x -> logdensityof(unshaped(PosteriorMeasure(ℒ ∘ reparam, prior)), x)
        @test Zygote.gradient(f_logd, x)[1] ≈ ForwardDiff.gradient(f_logd, x)
    end
end
