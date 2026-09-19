# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, Distributions, MeasureBase, ValueShapes
using ForwardDiff, Random
using MeasureBase: massof, superpose, StdNormal

@testset "superpos_measure" begin
    context = BATContext()

    m1 = batmeasure(Normal(0.0, 1.0))
    m2 = batmeasure(Normal(3.0, 0.5))

    s = 2.0 * m1 + 3.0 * m2
    @test s == superpose(2.0 * m1, 3.0 * m2)
    @test superpose(s) === s

    v = 1.3
    logd_ref = log(2.0 * pdf(Normal(0.0, 1.0), v) + 3.0 * pdf(Normal(3.0, 0.5), v))
    @test logdensityof(s, v) ≈ logd_ref
    @test BAT.checked_logdensityof(s, v) ≈ logd_ref
    @test float(massof(s)) ≈ 5.0

    @test logdensityof(m1 + m1, v) ≈ log(2.0) + logdensityof(m1, v)

    # Components with -Inf log-density are handled gracefully:
    m4 = batmeasure(Uniform(0.0, 1.0))
    @test logdensityof(m1 + m4, -1.0) ≈ logdensityof(m1, -1.0)

    @test varshape(s) == varshape(m1)

    # Sampling draws from the mass-weighted mixture of the components:
    @test BAT.supports_rand(s)
    @test rand(Random.default_rng(), s) isa Real
    smpls = bat_sample(s, IIDSampling(nsamples = 10^4), context).result
    mix_ref = MixtureModel([Normal(0.0, 1.0), Normal(3.0, 0.5)], [0.4, 0.6])
    @test BAT.test_dist_samples(mix_ref, smpls, context)

    # Zero-mass components must not be drawn from:
    s_zw = 0.0 * batmeasure(Uniform(0.0, 1.0)) + 2.0 * batmeasure(Uniform(5.0, 6.0))
    @test all(x -> 5.0 <= x <= 6.0, [rand(Random.default_rng(), s_zw) for _ in 1:100])

    # Log-density is differentiation-friendly:
    f = x -> logdensityof(2.0 * batmeasure(Normal(x, 1.0)) + 3.0 * m2, v)
    @test ForwardDiff.derivative(f, 0.0) ≈ ForwardDiff.derivative(
        x -> log(2.0 * pdf(Normal(x, 1.0), v) + 3.0 * pdf(Normal(3.0, 0.5), v)), 0.0)

    # Derivatives must be correct at equal log-densities, log(2 cosh(x)) has
    # derivative 0 and second derivative 1 at x = 0:
    for f_tie in (x -> BAT._logaddexp(1 - x, 1 + x), x -> BAT._logaddexp(1 + x, 1 - x))
        @test f_tie(0.0) ≈ 1 + log(2.0)
        @test abs(ForwardDiff.derivative(f_tie, 0.0)) < 1e-14
        @test ForwardDiff.derivative(x -> ForwardDiff.derivative(f_tie, x), 0.0) ≈ 1.0
    end
    # Also through the log-density of a superposition, at the crossing point
    # of two different components, where symmetry demands derivative 0:
    for s_cross in (batmeasure(Normal(-1.0, 1.0)) + batmeasure(Normal(1.0, 1.0)),
                    batmeasure(Normal(1.0, 1.0)) + batmeasure(Normal(-1.0, 1.0)))
        @test abs(ForwardDiff.derivative(x -> logdensityof(s_cross, x), 0.0)) < 1e-14
    end
end
