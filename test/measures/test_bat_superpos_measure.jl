# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test

using DensityInterface, Distributions, MeasureBase, ValueShapes
using ForwardDiff, Random
using MeasureBase: massof, superpose, StdNormal

@testset "bat_superpos_measure" begin
    context = BATContext()

    m1 = batmeasure(Normal(0.0, 1.0))
    m2 = batmeasure(Normal(3.0, 0.5))

    s = 2.0 * m1 + 3.0 * m2
    @test s isa BAT.BATSuperpositionMeasure
    @test @inferred(superpose(2.0 * m1, 3.0 * m2)) == s
    @test @inferred(superpose(s)) === s

    v = 1.3
    logd_ref = log(2.0 * pdf(Normal(0.0, 1.0), v) + 3.0 * pdf(Normal(3.0, 0.5), v))
    @test @inferred(logdensityof(s, v)) ≈ logd_ref
    @test @inferred(BAT.checked_logdensityof(s, v)) ≈ logd_ref
    @test float(massof(s)) ≈ 5.0

    # Superpositions flatten, equal measures don't collapse (type stability):
    @test (s + 4.0 * m1).components isa Tuple{Vararg{BAT.BATMeasure,3}}
    @test @inferred(m1 + m1) isa BAT.BATSuperpositionMeasure
    @test logdensityof(m1 + m1, v) ≈ log(2.0) + logdensityof(m1, v)

    # Vararg superpositions of BAT measures stay in the BATMeasure universe:
    m3 = batmeasure(Exponential(1.0))
    s3 = @inferred(superpose(m1, m2, m3))
    @test s3 isa BAT.BATSuperpositionMeasure
    @test length(s3.components) == 3
    @test logdensityof(s3, v) ≈ log(pdf(Normal(0.0, 1.0), v) + pdf(Normal(3.0, 0.5), v) + pdf(Exponential(1.0), v))
    @test superpose(m1, m2, s3).components isa Tuple{Vararg{BAT.BATMeasure,5}}

    # Mixing BAT measures with other measure types requires explicit
    # conversion instead of producing surprises:
    @test_throws ArgumentError superpose(m1, StdNormal())
    @test_throws ArgumentError m1 + StdNormal()
    @test_throws ArgumentError superpose(m1, m2, StdNormal())
    @test superpose(m1, batmeasure(StdNormal())) isa BAT.BATSuperpositionMeasure

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

    # Components that don't support rand:
    posterior = PosteriorMeasure(logfuncdensity(logdensityof(Normal())), Normal())
    s_nomass = m1 + posterior
    @test !BAT.supports_rand(s_nomass)
    @test_throws ArgumentError rand(Random.default_rng(), s_nomass)

    # Zero-mass components must not be selected, even at the u == 0 boundary:
    s_zw = 0.0 * batmeasure(Uniform(0.0, 1.0)) + 2.0 * batmeasure(Uniform(5.0, 6.0))
    w_zw = map(c -> float(massof(c)), s_zw.components)
    gen = BAT.get_gencontext(BATContext())
    @test 5.0 <= BAT._rand_superpos_component(gen, s_zw.components, w_zw, 0.0) <= 6.0
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
