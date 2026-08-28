# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using LinearAlgebra, Random, Statistics, StatsBase
using Distributions, ValueShapes, DensityInterface
using Random123
import ForwardDiff

# Statistical behavior of the native HMC stack on difficult target
# geometries (a fast subset of benchmark/hmc_geometry_benchmark.jl):
@testset "hmc_hard_targets" begin
    context = BATContext(rng = Philox4x((564, 81)), ad = ForwardDiff)

    function divergence_fraction(em)
        diags = BAT.evalinfo(em).result.chain_diagnostics
        sum(d.n_divergent for d in diags) / sum(d.n_transitions for d in diags)
    end

    @testset "ill-conditioned Gaussian" begin
        # Scales spanning three orders of magnitude; the transform tuner
        # has to supply the geometry (no pretransform help):
        d = 8
        σs = exp10.(range(-1, 2, length = d))
        objective = MvNormal(zeros(d), Diagonal(σs .^ 2))
        alg = TransformedMCMC(proposal = HamiltonianMC(), pretransform = DoNotTransform(), nsteps = 10^4)
        em = evalmeasure(batmeasure(objective), alg, context)
        @test BAT.test_dist_samples(objective, BAT.samplesof(em), context)
        @test divergence_fraction(em) < 0.05
    end

    @testset "strongly correlated Gaussian" begin
        d = 6
        Σ = Matrix(Symmetric([0.95^abs(i - j) * 1.0 for i in 1:d, j in 1:d]))
        objective = MvNormal(fill(2.0, d), Σ)
        alg = TransformedMCMC(proposal = HamiltonianMC(), pretransform = DoNotTransform(), nsteps = 10^4)
        em = evalmeasure(batmeasure(objective), alg, context)
        @test BAT.test_dist_samples(objective, BAT.samplesof(em), context)
        @test divergence_fraction(em) < 0.05
    end

    @testset "mild funnel" begin
        # A mild Neal-type funnel: v ~ N(0, 1.5), x_i | v ~ N(0, e^(v/2)).
        # Constant-metric HMC cannot resolve the neck perfectly, so only
        # the exactly known v-marginal is checked, with generous
        # tolerances, plus bounded trajectory pathology:
        prior = distprod(v = Normal(0.0, 1.5), x = MvNormal(zeros(3), I))
        loglik = logfuncdensity(
            p -> sum(logpdf.(Normal.(0.0, exp(p.v / 2)), p.x)) - sum(logpdf.(Normal(0.0, 1.0), p.x))
        )
        funnel = PosteriorMeasure(loglik, prior)
        alg = TransformedMCMC(proposal = HamiltonianMC(), nsteps = 2 * 10^4, strict = false)
        em = evalmeasure(funnel, alg, context)
        smpls = BAT.samplesof(em)
        vs = [smpl.v for smpl in smpls.v]
        w = Weights(smpls.weight)
        @test abs(mean(vs, w)) < 0.45
        @test 1.0 < std(vs, w, corrected = false) < 2.0
        @test divergence_fraction(em) < 0.2
    end
end
